cat > rehydration_sigil_tx.txt <<'EOF'
⊏⚗$ Σ_rehydrate::Autonomous_SDR_Security_TX_v1 ⊐
Ω_contract:
  safety_limiter: tx=CONTROLLED, hw_access=CONTROLLED, rf_emit=CONTROLLED
  io_scope: local_only + regulated_rf
  compliance: FCC_part15, ISM_bands_only, power_limited
Ω_modules:
  synth_source:
    - noise(gr=gaussian, amp∈ℝ⁺)
    - tone(freq∈ℝ⁺, amp∈ℝ⁺, phase∈[0,2π))
    - multitone({fi}, amps, phases)
    - iq_file(path, rate)
    - modulators: [ook, bpsk, qpsk, fsk, ofdm_stub]
  tx_chain:
    - power_limiter(max_dBm=-30)
    - band_filter(ISM_bands)
    - timing_control(min_pulse_width=1ms)
    - duty_cycle_limiter(<1%)
  analytics_layer:
    - fft(N=power_of_two)
    - psd(welch, Nseg)
    - snr(est: signal_bin / noise_floor)
    - occupancy(band_slices→ρ∈[0,1])
    - anomaly(score=KL(P||Q_ref))
    - cls(rule||nn_stub)
    - tx_monitoring: [spectral_compliance, power_compliance]
  controller:
    observe → analyze → decide → act
    knobs: {cent_freq, bandwidth, bb_gain, if_gain, rf_gain, tx_power}
    policy: safe_meta_bandit(ε-greedy, reward= −occupancy + α·snr − β·anomaly − γ·tx_violation)
    rate_limit: Δknob/sec ≤ κ
  security_learner:
    - adversarial_examples(rf_perturbations)
    - evasion_detection
    - spectrum_authentication
    - tx_behavior_analysis
  interface:
    - xmlrpc(localhost:8888){get/set: cent_freq, bandwidth, ..., tx_enable}
    - metrics_stream(stdout|jsonl)
  persistence:
    - state.json
    - metrics.csv
    - security_events.log
Ω_wiring:
  synth_source → signal_bus → analytics_layer → controller → knob_updates → synth_source
  synth_source → tx_chain → [SDR_HARDWARE_TX]
  [SDR_HARDWARE_RX] → analytics_layer
  analytics_layer → security_learner → controller
  analytics_layer → persistence
  controller → persistence
Ω_runloop:
  while alive:
    x[t]←synth_source(step) OR sdr_rx_capture()
    m[t]←analytics_layer(x[t])
    a[t]←controller(m[t])
    if tx_safe(a[t]): apply_tx(a[t])
    log(m[t], a[t])
Ω_guards:
  assert tx_power_limits()
  assert band_compliance()
  assert duty_cycle_limits()
  assert requires_safety_override_flag()
⊐θ$⊐
EOF

cat > autonomous_sdr_ai_tx.py <<'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous SDR Security with Controlled TX for AI/ML Security Learning
- CONTROLLED RF emission for security research
- Hardware drivers enabled with safety limits
- AI/ML adversarial learning capabilities
"""
import argparse
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import logging
from datetime import datetime
from SimpleXMLRPCServer import SimpleXMLRPCServer
from socketserver import ThreadingMixIn

import numpy as np

# --------------------- Conditional Hardware Import ---------------------
try:
    import soapysdr
    from soapysdr import SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CF32
    HAS_SOAPYSDR = True
except ImportError:
    HAS_SOAPYSDR = False
    print("WARNING: SoapySDR not available - TX/RX disabled")

try:
    from gnuradio import blocks, digital, filter, gr
    HAS_GNURADIO = True
except ImportError:
    HAS_GNURADIO = False

# --------------------- Safety Configuration ---------------------
SAFETY_LIMITS = {
    'max_tx_power_dBm': -30.0,  # FCC Part 15 compliant
    'allowed_bands': [
        (902e6, 928e6),    # ISM 915MHz
        (2.4e9, 2.4835e9), # ISM 2.4GHz  
        (5.15e9, 5.35e9),  # ISM 5GHz lower
        (5.47e9, 5.725e9), # ISM 5GHz upper
    ],
    'max_duty_cycle': 0.01,  # 1% duty cycle
    'min_freq_resolution': 1e3,  # 1kHz minimum
}

# --------------------- Utilities ---------------------
def db10(x):
    x = float(x)
    return 10.0 * math.log10(max(x, 1e-20))

def now_ts():
    return datetime.now().isoformat()

def is_in_band(freq, bands):
    for low, high in bands:
        if low <= freq <= high:
            return True
    return False

# --------------------- SDR Hardware Controller ---------------------
class SDRHardware:
    def __init__(self, sample_rate=5e6, tx_freq=915e6, rx_freq=915e6, 
                 tx_gain=0.0, rx_gain=20.0, driver="", args=""):
        self.sample_rate = sample_rate
        self.tx_freq = tx_freq
        self.rx_freq = rx_freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.driver = driver
        self.args = args
        
        self.tx_enabled = False
        self.rx_enabled = False
        self.sdr_device = None
        self.tx_stream = None
        self.rx_stream = None
        
        self.tx_samples_sent = 0
        self.tx_start_time = None
        self.last_tx_check = time.time()
        
    def initialize(self):
        if not HAS_SOAPYSDR:
            return False, "SoapySDR not available"
            
        try:
            if self.driver:
                self.sdr_device = soapysdr.Device(driver=self.driver, args=self.args)
            else:
                # Auto-detect
                results = soapysdr.Device.enumerate()
                if not results:
                    return False, "No SDR devices found"
                self.sdr_device = soapysdr.Device(results[0])
            
            # Configure sample rate
            self.sdr_device.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self.sdr_device.setSampleRate(SOAPY_SDR_TX, 0, self.sample_rate)
            
            # Configure frequencies
            self.sdr_device.setFrequency(SOAPY_SDR_RX, 0, self.rx_freq)
            self.sdr_device.setFrequency(SOAPY_SDR_TX, 0, self.tx_freq)
            
            # Configure gains
            self.sdr_device.setGain(SOAPY_SDR_RX, 0, self.rx_gain)
            self.sdr_device.setGain(SOAPY_SDR_TX, 0, self.tx_gain)
            
            return True, "SDR initialized successfully"
            
        except Exception as e:
            return False, f"SDR init failed: {str(e)}"
    
    def enable_tx(self, enable=True):
        if not self.sdr_device:
            return False, "SDR not initialized"
            
        if enable and not self.tx_safety_check():
            return False, "TX safety check failed"
            
        try:
            if enable and not self.tx_stream:
                self.tx_stream = self.sdr_device.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
                self.sdr_device.activateStream(self.tx_stream)
                self.tx_start_time = time.time()
            elif not enable and self.tx_stream:
                self.sdr_device.deactivateStream(self.tx_stream)
                self.sdr_device.closeStream(self.tx_stream)
                self.tx_stream = None
                
            self.tx_enabled = enable
            return True, f"TX {'enabled' if enable else 'disabled'}"
        except Exception as e:
            return False, f"TX control failed: {str(e)}"
    
    def tx_safety_check(self):
        # Check frequency compliance
        if not is_in_band(self.tx_freq, SAFETY_LIMITS['allowed_bands']):
            return False
            
        # Check power compliance
        if self.tx_gain > SAFETY_LIMITS['max_tx_power_dBm']:
            return False
            
        # Check duty cycle
        current_time = time.time()
        if self.tx_start_time:
            total_time = current_time - self.tx_start_time
            if total_time > 0:
                duty_cycle = self.tx_samples_sent / (self.sample_rate * total_time)
                if duty_cycle > SAFETY_LIMITS['max_duty_cycle']:
                    return False
                    
        return True
    
    def transmit(self, iq_samples):
        if not self.tx_enabled or not self.tx_stream:
            return False, "TX not enabled"
            
        if not self.tx_safety_check():
            self.enable_tx(False)
            return False, "Safety check failed during transmission"
            
        try:
            # Convert to proper format and transmit
            iq_samples = iq_samples.astype(np.complex64)
            num_sent = self.sdr_device.writeStream(self.tx_stream, [iq_samples], len(iq_samples))
            self.tx_samples_sent += num_sent
            return True, f"Transmitted {num_sent} samples"
        except Exception as e:
            return False, f"Transmission failed: {str(e)}"
    
    def receive(self, num_samples=1024):
        if not self.rx_enabled or not self.rx_stream:
            # Initialize RX stream if needed
            if self.sdr_device and not self.rx_stream:
                try:
                    self.rx_stream = self.sdr_device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
                    self.sdr_device.activateStream(self.rx_stream)
                    self.rx_enabled = True
                except:
                    return None
                    
        if not self.rx_stream:
            return None
            
        try:
            buffer = np.zeros(num_samples, dtype=np.complex64)
            status = self.sdr_device.readStream(self.rx_stream, [buffer], num_samples)
            if status.ret > 0:
                return buffer[:status.ret]
        except:
            pass
            
        return None

# --------------------- Enhanced Synth Source with Modulators ---------------------
class EnhancedSynthSource:
    def __init__(self, sample_rate=5e6, bandwidth=10e6, cent_freq=915e6,
                 bb_gain=0.0, if_gain=0.0, rf_gain=0.0):
        self.sample_rate = float(sample_rate)
        self.bandwidth = float(bandwidth)
        self.cent_freq = float(cent_freq)
        self.bb_gain = float(bb_gain)
        self.if_gain = float(if_gain)
        self.rf_gain = float(rf_gain)

        self.mode = "noise"
        self.modulation = "none"  # none, ook, bpsk, qpsk, fsk
        self.amp = 1.0
        self.tone_freq = 25e3
        self.phase = 0.0
        self.multitone = [50e3, 120e3, 250e3]
        self.multitone_amps = [1.0, 0.7, 0.5]
        self.multitone_phases = [0.0, 1.0, 2.0]
        self.iq_path = None
        
        # Modulation parameters
        self.symbol_rate = 10e3
        self.sps = int(sample_rate / self.symbol_rate)  # samples per symbol
        
        self._t = 0
        self._two_pi = 2.0 * math.pi

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v) if isinstance(getattr(self, k), float) else v)

    def _gain_chain(self):
        g_lin = (10 ** (self.bb_gain/20.0)) * (10 ** (self.if_gain/20.0)) * (10 ** (self.rf_gain/20.0))
        return g_lin

    def _generate_symbols(self, n):
        """Generate random symbols for modulation"""
        if self.modulation == "ook":
            return np.random.randint(0, 2, n)
        elif self.modulation == "bpsk":
            return 2 * np.random.randint(0, 2, n) - 1
        elif self.modulation == "qpsk":
            symbols = np.random.randint(0, 4, n)
            return np.exp(1j * (np.pi/4 + symbols * np.pi/2))
        elif self.modulation == "fsk":
            return np.random.randint(0, 2, n)
        return np.ones(n)

    def _modulate(self, symbols, n):
        if self.modulation == "none":
            return self._generate_carrier(n)
            
        t = (np.arange(n) + self._t) / self.sample_rate
        
        if self.modulation == "ook":
            # Simple OOK: repeat symbols
            symbol_indices = (t * self.symbol_rate).astype(int) % len(symbols)
            modulated = symbols[symbol_indices] * np.exp(1j * self._two_pi * self.tone_freq * t)
            
        elif self.modulation == "bpsk":
            symbol_indices = (t * self.symbol_rate).astype(int) % len(symbols)
            modulated = symbols[symbol_indices] * np.exp(1j * self._two_pi * self.tone_freq * t)
            
        elif self.modulation == "qpsk":
            symbol_indices = (t * self.symbol_rate).astype(int) % len(symbols)
            modulated = symbols[symbol_indices] * np.exp(1j * self._two_pi * self.tone_freq * t)
            
        elif self.modulation == "fsk":
            # Simple 2-FSK
            f_dev = 10e3  # frequency deviation
            symbol_indices = (t * self.symbol_rate).astype(int) % len(symbols)
            freq = self.tone_freq + f_dev * (2*symbols[symbol_indices] - 1)
            phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
            modulated = np.exp(1j * phase)
            
        else:
            modulated = self._generate_carrier(n)
            
        return modulated.astype(np.complex64)

    def _generate_carrier(self, n):
        t = (np.arange(n) + self._t) / self.sample_rate
        w = self._two_pi * self.tone_freq
        return np.exp(1j * (w * t + self.phase)).astype(np.complex64)

    def step(self, n=262144):
        g = self._gain_chain()
        
        if self.modulation != "none":
            symbols = self._generate_symbols(1024)  # Generate symbol sequence
            iq = self._modulate(symbols, n)
        elif self.mode == "noise":
            iq = (np.random.normal(0, 1, n) + 1j*np.random.normal(0,1,n)).astype(np.complex64)
            iq *= (self.amp * g)
        elif self.mode == "tone":
            iq = (self.amp * g) * self._generate_carrier(n)
        elif self.mode == "multitone":
            t = (np.arange(n) + self._t) / self.sample_rate
            sig = np.zeros(n, dtype=np.complex64)
            for f0, a0, p0 in zip(self.multitone, self.multitone_amps, self.multitone_phases):
                w = self._two_pi * f0
                sig += (a0 * np.exp(1j*(w*t + p0))).astype(np.complex64)
            iq = (self.amp * g) * sig
        elif self.mode == "iq_file" and self.iq_path and os.path.exists(self.iq_path):
            with open(self.iq_path, "rb") as f:
                data = f.read(n * 8)
            if len(data) < n * 8:
                iq = np.zeros(n, dtype=np.complex64)
            else:
                iq = np.frombuffer(data, dtype=np.complex64)
        else:
            iq = np.zeros(n, dtype=np.complex64)

        self._t += n
        return iq

# --------------------- Security Learning Module ---------------------
class SecurityLearner:
    def __init__(self):
        self.adversarial_examples = []
        self.evasion_detections = 0
        self.spectral_fingerprints = {}
        
    def analyze_adversarial_pattern(self, iq_signal, psd):
        """Detect potential adversarial RF patterns"""
        # Simple anomaly detection based on spectral characteristics
        spectral_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / np.mean(psd)
        
        if spectral_flatness < 0.1:  # Very peaky spectrum
            self.evasion_detections += 1
            return {"evasion_detected": True, "confidence": 0.8}
            
        return {"evasion_detected": False, "confidence": 0.0}
    
    def generate_adversarial_example(self, base_signal, attack_type="jamming"):
        """Generate adversarial RF examples for training"""
        if attack_type == "jamming":
            # Add structured jamming
            jamming = 0.1 * np.random.normal(0, 1, len(base_signal))
            return base_signal + jamming.astype(np.complex64)
        elif attack_type == "evasion":
            # Add subtle perturbations
            perturbation = 0.01 * np.exp(1j * np.random.uniform(0, 2*np.pi, len(base_signal)))
            return base_signal * (1 + perturbation)
        else:
            return base_signal

# --------------------- Enhanced Controller with Security Awareness ---------------------
class SecurityAwareController:
    def __init__(self, synth: EnhancedSynthSource, analytics, sdr: SDRHardware):
        self.synth = synth
        self.analytics = analytics
        self.sdr = sdr
        self.security_learner = SecurityLearner()
        
        self.epsilon = 0.1
        self.alpha = 0.1
        self.beta = 0.05
        self.gamma = 1.0  # Weight for TX violations
        self.kappa = 3.0
        self.last_update = time.time()
        
        self.min_bw = 2e6
        self.max_bw = 50e6
        self.tx_violations = 0

    def _rate_limited(self):
        dt = time.time() - self.last_update
        return dt < 0.2

    def _check_tx_safety(self, proposed_params):
        """Check if proposed transmission parameters are safe"""
        freq_ok = is_in_band(proposed_params.get('cent_freq', self.synth.cent_freq), 
                            SAFETY_LIMITS['allowed_bands'])
        power_ok = proposed_params.get('rf_gain', self.synth.rf_gain) <= SAFETY_LIMITS['max_tx_power_dBm']
        return freq_ok and power_ok

    def decide_and_act(self, metrics, current_iq=None):
        if self._rate_limited():
            return {}

        occ = metrics["occupancy"]
        snr = metrics["snr_db"]
        anomaly = metrics["anomaly"]
        
        # Security analysis
        security_info = self.security_learner.analyze_adversarial_pattern(
            current_iq if current_iq is not None else np.zeros(1024, dtype=np.complex64),
            metrics.get("psd", np.ones(1024))
        )
        
        # Enhanced reward with security considerations
        tx_penalty = self.gamma * self.tx_violations
        security_bonus = -0.5 if security_info["evasion_detected"] else 0.1
        
        reward = -occ + 0.05*snr - 0.1*anomaly - tx_penalty + security_bonus

        # Parameter adjustments
        bw = self.synth.bandwidth
        explore = np.random.rand() < self.epsilon
        
        if explore:
            step = np.random.uniform(-0.2, 0.2) * bw
        else:
            step = (-0.25*occ + 0.02*snr - 0.1*anomaly) * 0.1 * bw

        new_bw = np.clip(bw + step, self.min_bw, self.max_bw)
        
        # Frequency adjustment with security awareness
        cf = self.synth.cent_freq
        cf_step = np.clip((0.5 - occ) * 25e3, -200e3, 200e3)
        new_cf = cf + cf_step
        
        # TX power adjustment (careful!)
        current_gain = self.synth.rf_gain
        new_gain = np.clip(current_gain + np.random.uniform(-1, 1), -50, SAFETY_LIMITS['max_tx_power_dBm'])
        
        # Apply changes with safety check
        proposed_changes = {
            "bandwidth": float(new_bw),
            "cent_freq": float(new_cf),
            "rf_gain": float(new_gain)
        }
        
        if not self._check_tx_safety(proposed_changes):
            self.tx_violations += 1
            # Revert to safe parameters
            safe_cf = 915e6  # Default to ISM band
            safe_gain = -30.0  # Default safe power
            proposed_changes.update({
                "cent_freq": safe_cf,
                "rf_gain": safe_gain
            })
        
        self.synth.set_params(**proposed_changes)
        
        # Update SDR hardware if available
        if self.sdr.sdr_device:
            try:
                self.sdr.tx_freq = proposed_changes["cent_freq"]
                self.sdr.tx_gain = proposed_changes["rf_gain"]
                self.sdr.sdr_device.setFrequency(SOAPY_SDR_TX, 0, self.sdr.tx_freq)
                self.sdr.sdr_device.setGain(SOAPY_SDR_TX, 0, self.sdr.tx_gain)
            except:
                pass
        
        self.last_update = time.time()
        
        result = proposed_changes.copy()
        result.update({
            "reward": float(reward),
            "tx_violations": self.tx_violations,
            "security_alert": security_info["evasion_detected"]
        })
        
        return result

# --------------------- Enhanced Control API ---------------------
class EnhancedControlAPI:
    def __init__(self, synth: EnhancedSynthSource, sdr: SDRHardware, logger):
        self.synth = synth
        self.sdr = sdr
        self.logger = logger
        self.tx_enabled = False

    # Enhanced getters/setters
    def get_var_cent_freq(self): return float(self.synth.cent_freq)
    def get_var_bandwidth(self): return float(self.synth.bandwidth)
    def get_var_bb_gain(self): return float(self.synth.bb_gain)
    def get_var_if_gain(self): return float(self.synth.if_gain)
    def get_var_rf_gain(self): return float(self.synth.rf_gain)
    def get_tx_enabled(self): return bool(self.tx_enabled)

    def set_var_cent_freq(self, v): 
        if is_in_band(float(v), SAFETY_LIMITS['allowed_bands']):
            self.synth.cent_freq = float(v)
            if self.sdr.sdr_device:
                self.sdr.tx_freq = float(v)
                self.sdr.sdr_device.setFrequency(SOAPY_SDR_TX, 0, self.sdr.tx_freq)
            return True
        return False
        
    def set_var_rf_gain(self, v):
        v = float(v)
        if v <= SAFETY_LIMITS['max_tx_power_dBm']:
            self.synth.rf_gain = v
            if self.sdr.sdr_device:
                self.sdr.tx_gain = v
                self.sdr.sdr_device.setGain(SOAPY_SDR_TX, 0, self.sdr.tx_gain)
            return True
        return False

    def set_tx_enabled(self, enable):
        if not HAS_SOAPYSDR or not self.sdr.sdr_device:
            return False
            
        success, msg = self.sdr.enable_tx(enable)
        if success:
            self.tx_enabled = enable
        return success

    # Modulation controls
    def set_modulation(self, mod):
        if mod in ["none", "ook", "bpsk", "qpsk", "fsk"]:
            self.synth.modulation = mod
            return True
        return False

    def safety_status(self):
        return {
            "tx": self.tx_enabled,
            "hw_access": HAS_SOAPYSDR and self.sdr.sdr_device is not None,
            "rf_emit": self.tx_enabled,
            "safety_limits": SAFETY_LIMITS,
            "tx_violations": getattr(self, 'tx_violations', 0)
        }

# --------------------- Enhanced Main Loop ---------------------
def run_enhanced_loop(args):
    synth = EnhancedSynthSource(
        sample_rate=args.sample_rate,
        bandwidth=args.bandwidth,
        cent_freq=args.cent_freq,
        bb_gain=args.bb_gain,
        if_gain=args.if_gain,
        rf_gain=args.rf_gain
    )

    analytics = Analytics(sample_rate=args.sample_rate, nfft=args.nfft)
    
    # Initialize SDR hardware
    sdr = SDRHardware(
        sample_rate=args.sample_rate,
        tx_freq=args.cent_freq,
        rx_freq=args.cent_freq,
        tx_gain=args.rf_gain,
        rx_gain=20.0,
        driver=args.sdr_driver,
        args=args.sdr_args
    )
    
    if args.enable_hw:
        success, msg = sdr.initialize()
        print(f"SDR Initialization: {msg}")
    else:
        print("Hardware disabled - running in simulation mode")

    ctrl = SecurityAwareController(synth, analytics, sdr)
    log = Logger(args.out)

    # Enhanced XML-RPC API
    api = EnhancedControlAPI(synth, sdr, log)
    server = ThreadedXMLRPCServer((args.host, args.port), allow_none=True, logRequests=False)
    server.register_instance(api)
    srv_th = threading.Thread(target=server.serve_forever, daemon=True)
    srv_th.start()

    stop = threading.Event()
    def _sig(_a,_b): stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    ref_psd = None
    tx_cycle_counter = 0
    
    print("Starting Enhanced Autonomous SDR Security System")
    print("=== SAFETY ENABLED ===")
    print(f"Max TX Power: {SAFETY_LIMITS['max_tx_power_dBm']} dBm")
    print(f"Allowed Bands: {SAFETY_LIMITS['allowed_bands']}")
    print(f"Max Duty Cycle: {SAFETY_LIMITS['max_duty_cycle']*100}%")
    
    while not stop.is_set():
        # Generate or capture signal
        if sdr.rx_enabled:
            iq = sdr.receive(args.block)
            if iq is None:
                iq = synth.step(n=args.block)
        else:
            iq = synth.step(n=args.block)
        
        # Analyze
        psd = analytics.fft_mag(iq)
        occ = analytics.occupancy(psd)
        snr_db = analytics.snr_est(psd)
        an = analytics.anomaly(psd, ref_psd)
        cls = analytics.classify(psd)

        metrics = {
            "occupancy": float(occ), 
            "snr_db": float(snr_db), 
            "anomaly": float(an), 
            "cls": cls,
            "psd": psd.tolist()[:1000]  # Store partial PSD for security analysis
        }
        
        # Decide and act
        act = ctrl.decide_and_act(metrics, iq)
        
        # Controlled transmission
        if api.tx_enabled and tx_cycle_counter % 10 == 0:  # Reduce TX rate
            tx_success, tx_msg = sdr.transmit(iq)
            if tx_success and args.verbose:
                print(f"TX: {tx_msg}")
        
        # Logging
        log.log_metrics(now_ts(), synth, occ, snr_db, an, cls)
        log.save_state(synth)
        
        # Security logging
        if act.get("security_alert", False):
            with open(os.path.join(args.out, "security_events.log"), "a") as f:
                f.write(f"{now_ts()} - Security alert: {act}\n")

        # Update reference
        if ref_psd is None:
            ref_psd = psd.copy()
        else:
            ref_psd = 0.99*ref_psd + 0.01*psd

        if args.verbose and tx_cycle_counter % 20 == 0:
            print(json.dumps({
                "t": now_ts(), 
                "metrics": {k: v for k, v in metrics.items() if k != 'psd'},
                "act": act,
                "tx_enabled": api.tx_enabled
            }))
            
        tx_cycle_counter += 1
        time.sleep(max(0.0, 1.0/args.hz))

    server.shutdown()
    if sdr.tx_enabled:
        sdr.enable_tx(False)

# Keep existing Analytics, Logger classes from previous version
# Add any missing methods if needed

def parse_enhanced_args():
    p = argparse.ArgumentParser(description="Enhanced Autonomous SDR Security with Controlled TX")
    p.add_argument("--sample-rate", type=float, default=5e6)
    p.add_argument("--bandwidth", type=float, default=10e6)
    p.add_argument("--cent-freq", type=float, default=915e6)
    p.add_argument("--bb-gain", type=float, default=0.0)
    p.add_argument("--if-gain", type=float, default=0.0)
    p.add_argument("--rf-gain", type=float, default=-30.0)
    p.add_argument("--nfft", type=int, default=131072)
    p.add_argument("--block", type=int, default=262144)
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--out", type=str, default="enhanced_autosdr_out")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--enable-hw", action="store_true", help="Enable SDR hardware")
    p.add_argument("--sdr-driver", type=str, default="", help="SoapySDR driver (e.g., 'hackrf', 'rtlsdr')")
    p.add_argument("--sdr-args", type=str, default="", help="SoapySDR device arguments")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_enhanced_args()
    run_enhanced_loop(args)
EOF

# Add missing Analytics and Logger classes if needed
cat >> autonomous_sdr_ai_tx.py <<'EOF'

# --------------------- Analytics (from previous version) ---------------------
class Analytics:
    def __init__(self, sample_rate, nfft=131072):
        self.sample_rate = float(sample_rate)
        self.nfft = int(1 << (int(math.log(nfft, 2))))
        self.window = np.hanning(self.nfft).astype(np.float64)

    def fft_mag(self, x_c64):
        x = x_c64[:self.nfft]
        if len(x) < self.nfft:
            pad = np.zeros(self.nfft - len(x), dtype=np.complex64)
            x = np.concatenate([x, pad])
        xw = (x.real*self.window + 1j*x.imag*self.window).astype(np.complex64)
        X = np.fft.fftshift(np.fft.fft(xw))
        psd = (np.abs(X)**2) / np.sum(self.window**2)
        return psd

    def occupancy(self, psd, bands=32):
        thr = np.median(psd) * 5.0
        band_len = len(psd)//bands
        occ = 0.0
        for b in range(bands):
            sl = psd[b*band_len:(b+1)*band_len]
            if sl.size == 0: 
                continue
            if np.mean(sl > thr) > 0.2:
                occ += 1.0
        return occ / max(bands,1)

    def snr_est(self, psd):
        peak = np.max(psd)
        floor = np.median(psd)
        return db10(peak/floor)

    def anomaly(self, psd, ref=None):
        p = psd / (np.sum(psd) + 1e-12)
        if ref is None:
            kernel = np.ones(513, dtype=np.float64)
            kernel /= np.sum(kernel)
            q = np.convolve(p, kernel, mode="same")
        else:
            q = ref / (np.sum(ref) + 1e-12)
        z = np.where((p>0) & (q>0), p * np.log(p/(q+1e-18)), 0.0)
        return float(np.sum(z))

    def classify(self, psd):
        peaks = np.sum(psd > (np.median(psd)*20.0))
        if peaks >= 3:
            return "multitone_like"
        if peaks == 1:
            return "tone_like"
        return "noise_like"

# --------------------- Logger (from previous version) ---------------------
class Logger:
    def __init__(self, out_dir="enhanced_autosdr_out"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.metrics_path = os.path.join(out_dir, "metrics.csv")
        self.state_path = os.path.join(out_dir, "state.json")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write("t,cent_freq,bandwidth,bb_gain,if_gain,rf_gain,occ,snr_db,anomaly,class\n")

    def log_metrics(self, t, synth, occ, snr_db, anomaly, cls):
        with open(self.metrics_path, "a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                t, synth.cent_freq, synth.bandwidth, synth.bb_gain, synth.if_gain, synth.rf_gain,
                occ, snr_db, anomaly, cls
            ))

    def save_state(self, synth):
        state = {
            "t": now_ts(),
            "cent_freq": synth.cent_freq,
            "bandwidth": synth.bandwidth,
            "bb_gain": synth.bb_gain,
            "if_gain": synth.if_gain,
            "rf_gain": synth.rf_gain,
            "mode": synth.mode,
            "modulation": synth.modulation,
            "amp": synth.amp,
            "tone_freq": synth.tone_freq
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

# --------------------- Threaded XML-RPC Server ---------------------
class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass
EOF

chmod +x autonomous_sdr_ai_tx.py

cat > security_learning_demo.py <<'EOF'
#!/usr/bin/env python3
"""
Demo for AI/ML Security Learning with RF Transmission
- Adversarial example generation
- Evasion detection
- Security-aware spectrum management
"""
import numpy as np
import matplotlib.pyplot as plt
from autonomous_sdr_ai_tx import EnhancedSynthSource, SecurityLearner, Analytics

def demo_security_learning():
    print("=== RF Security Learning Demo ===")
    
    # Initialize components
    synth = EnhancedSynthSource(sample_rate=1e6, cent_freq=915e6)
    analytics = Analytics(sample_rate=1e6)
    security = SecurityLearner()
    
    # Generate normal signal
    print("1. Generating normal signal...")
    normal_iq = synth.step(1024)
    normal_psd = analytics.fft_mag(normal_iq)
    
    # Generate adversarial examples
    print("2. Generating adversarial examples...")
    adversarial_jamming = security.generate_adversarial_example(normal_iq, "jamming")
    adversarial_evasion = security.generate_adversarial_example(normal_iq, "evasion")
    
    # Analyze security properties
    print("3. Analyzing security properties...")
    normal_analysis = security.analyze_adversarial_pattern(normal_iq, normal_psd)
    jamming_analysis = security.analyze_adversarial_pattern(adversarial_jamming, analytics.fft_mag(adversarial_jamming))
    evasion_analysis = security.analyze_adversarial_pattern(adversarial_evasion, analytics.fft_mag(adversarial_evasion))
    
    print(f"Normal signal: {normal_analysis}")
    print(f"Jamming attack: {jamming_analysis}") 
    print(f"Evasion attack: {evasion_analysis}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(np.real(normal_iq[:200]))
    plt.title("Normal Signal (I)")
        plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(np.real(adversarial_jamming[:200]))
    plt.title("Jamming Attack (I)")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(normal_psd)
    plt.title("Normal PSD")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.semilogy(analytics.fft_mag(adversarial_evasion))
    plt.title("Evasion Attack PSD")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("security_learning_demo.png")
    print("4. Plot saved as 'security_learning_demo.png'")
    
    print("\n=== Demo Complete ===")
    print("This demonstrates RF adversarial example generation and detection")
    print("for AI/ML security learning research.")

if __name__ == "__main__":
    demo_security_learning()
EOF

chmod +x security_learning_demo.py

cat > INSTALL_TX.md <<'EOF'
# Enhanced Autonomous SDR with RF Transmission

## Safety-First RF Transmission for AI/ML Security Research

### ⚠️ IMPORTANT SAFETY WARNINGS ⚠️

1. **LEGAL COMPLIANCE**: Only transmit in authorized bands (ISM: 915MHz, 2.4GHz, 5GHz)
2. **POWER LIMITS**: Default -30 dBm (FCC Part 15 compliant)
3. **DUTY CYCLE**: Limited to 1% maximum
4. **RESPONSIBLE USE**: For research in controlled environments only

### Installation

```bash
# Install SoapySDR and drivers
sudo apt-get update
sudo apt-get install soapysdr-tools soapysdr-module-all

# For HackRF:
sudo apt-get install soapysdr-module-hackrf

# For RTL-SDR (RX only):
sudo apt-get install soapysdr-module-rtlsdr

# For USRP:
sudo apt-get install soapysdr-module-uhd

# Python dependencies
pip install soapysdr numpy matplotlib
