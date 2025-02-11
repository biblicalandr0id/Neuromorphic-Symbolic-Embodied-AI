import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math

@dataclass
class IonChannelParams:
    """Parameters for voltage-gated ion channels"""
    Na_activation_slope: float = 0.1
    Na_activation_midpoint: float = -40.0
    Na_inactivation_slope: float = -0.056
    Na_inactivation_midpoint: float = -60.0
    K_activation_slope: float = 0.07
    K_activation_midpoint: float = -50.0
    g_Na_max: float = 120.0  # Maximum sodium conductance
    g_K_max: float = 36.0    # Maximum potassium conductance
    g_L: float = 0.3        # Leak conductance
    E_Na: float = 55.0      # Sodium reversal potential
    E_K: float = -72.0      # Potassium reversal potential
    E_L: float = -49.0      # Leak reversal potential

class BrainNetwork(nn.Module):
    def __init__(self):
        super(BrainNetwork, self).__init__()

        # Biophysical Constants
        self.Vrest = -70.0  # mV (Resting potential)
        self.Vpeak = 40.0   # mV (Peak potential)
        self.tau = 20.0     # ms (Time constant)
        self.theta = -55.0  # mV (Firing threshold)
        self.eta = 0.001    # Learning rate
        self.rho = 0.1      # Plasticity coefficient
        self.dt = 0.1       # ms (Time step)

        # Ion channel parameters
        self.ion_channels = IonChannelParams()

        # Enhanced calcium dynamics constants
        self.Ca_0 = 100.0      # nM (Baseline calcium)
        self.Ca_max = 1000.0   # nM (Maximum calcium)
        self.Ca_tau = 50.0     # ms (Calcium decay constant)
        self.Ca_threshold = -50.0  # mV (Voltage threshold for Ca influx)
        self.Ca_slope = 0.1    # Calcium voltage-dependence slope
        self.Ca_pump_max = 2.0 # Maximum pump rate
        self.Ca_pump_Kd = 0.1  # Pump dissociation constant

        # Enhanced energy parameters
        self.ATP_min = 1000.0   # μM (Minimum ATP required)
        self.ATP_0 = 5000.0     # μM (Initial ATP)
        self.ATP_max = 10000.0  # μM (Maximum ATP)
        self.ATP_synthesis_rate = 100.0  # μM/ms
        self.ATP_usage_per_spike = 1.0   # μM/spike
        self.glucose_concentration = 5.0  # mM
        self.oxygen_concentration = 160.0 # mmHg

        # Enhanced neurotransmitter parameters
        self.NT_types = {
            'glutamate': {'baseline': 1.0, 'max': 10.0, 'tau': 100.0},
            'GABA': {'baseline': 0.5, 'max': 5.0, 'tau': 50.0},
            'dopamine': {'baseline': 0.1, 'max': 2.0, 'tau': 200.0},
            'serotonin': {'baseline': 0.2, 'max': 3.0, 'tau': 150.0},
            'norepinephrine': {'baseline': 0.3, 'max': 4.0, 'tau': 180.0}
        }

        # STDP parameters
        self.stdp_window = 20.0  # ms
        self.stdp_A_plus = 0.1   # Maximum potentiation
        self.stdp_A_minus = -0.15  # Maximum depression
        self.stdp_tau_plus = 20.0  # Potentiation time constant
        self.stdp_tau_minus = 20.0  # Depression time constant

        # Homeostatic plasticity parameters
        self.target_firing_rate = 10.0  # Hz
        self.homeostatic_time_constant = 1000.0  # ms
        self.homeostatic_learning_rate = 0.01

        # Initialize weight matrices with biological constraints
        self.initialize_weights()

        # Neural layers with sparse connectivity
        self.sensory_layer = self.create_sparse_layer(6, 128, sparsity=0.8)
        self.primary_layer = self.create_sparse_layer(128, 256, sparsity=0.85)
        self.association_layer = self.create_sparse_layer(256, 512, sparsity=0.9)
        self.higher_layer = self.create_sparse_layer(512, 1024, sparsity=0.92)

        # Parallel processing pathways
        self.limbic_layer = self.create_sparse_layer(1024, 256, sparsity=0.9)
        self.basal_layer = self.create_sparse_layer(1024, 256, sparsity=0.9)
        self.cerebellum_layer = self.create_sparse_layer(1024, 256, sparsity=0.95)

        # Sequential processing layers
        self.memory_layer = self.create_sparse_layer(1024, 512, sparsity=0.9)
        self.executive_layer = self.create_sparse_layer(512, 256, sparsity=0.88)

        # Output layers
        self.motor_output = self.create_sparse_layer(256, 64, sparsity=0.8)
        self.autonomic_output = self.create_sparse_layer(256, 32, sparsity=0.8)
        self.cognitive_output = self.create_sparse_layer(256, 128, sparsity=0.85)

        # Initialize spike history
        self.spike_history = []
        self.max_history_length = 1000  # Maximum length of spike history

    def create_sparse_layer(self, in_features: int, out_features: int, 
                          sparsity: float) -> nn.Module:
        """Creates a sparse neural layer with controlled connectivity"""
        layer = nn.Linear(in_features, out_features)
        mask = torch.rand(out_features, in_features) > sparsity
        layer.weight.data *= mask.float()
        layer.weight.register_hook(lambda grad: grad * mask.float())
        return layer

    def initialize_weights(self):
        """Initialize weight matrices with biological constraints"""
        # Sensory weights initialization
        self.W_visual = self.create_biologically_constrained_weights(64, 64)
        self.W_auditory = self.create_biologically_constrained_weights(32, 32)
        self.W_tactile = self.create_biologically_constrained_weights(32, 32)
        self.W_olfactory = self.create_biologically_constrained_weights(16, 16)
        self.W_gustatory = self.create_biologically_constrained_weights(16, 16)
        self.W_proprioceptive = self.create_biologically_constrained_weights(32, 32)

    def create_biologically_constrained_weights(self, rows: int, cols: int) -> torch.Tensor:
        """Creates weight matrix with biological constraints"""
        # Initialize with log-normal distribution
        weights = torch.exp(torch.randn(rows, cols) * 0.4 - 0.5)
        
        # Apply Dale's Law: neurons are either excitatory or inhibitory
        is_excitatory = torch.rand(rows, 1) > 0.2  # 80% excitatory neurons
        weights = weights * is_excitatory
        
        # Scale weights to maintain network stability
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-6)
        
        return nn.Parameter(weights)

    def hodgkin_huxley_dynamics(self, V: torch.Tensor, m: torch.Tensor, 
                              h: torch.Tensor, n: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Implements Hodgkin-Huxley dynamics for more realistic action potentials
        """
        # Update gate variables
        alpha_m = 0.1 * (V + 40) / (1 - torch.exp(-(V + 40) / 10))
        beta_m = 4.0 * torch.exp(-(V + 65) / 18)
        alpha_h = 0.07 * torch.exp(-(V + 65) / 20)
        beta_h = 1.0 / (1 + torch.exp(-(V + 35) / 10))
        alpha_n = 0.01 * (V + 55) / (1 - torch.exp(-(V + 55) / 10))
        beta_n = 0.125 * torch.exp(-(V + 65) / 80)

        # Gate dynamics
        dm = alpha_m * (1 - m) - beta_m * m
        dh = alpha_h * (1 - h) - beta_h * h
        dn = alpha_n * (1 - n) - beta_n * n

        # Conductances
        g_Na = self.ion_channels.g_Na_max * m**3 * h
        g_K = self.ion_channels.g_K_max * n**4

        # Current calculation
        I_Na = g_Na * (V - self.ion_channels.E_Na)
        I_K = g_K * (V - self.ion_channels.E_K)
        I_L = self.ion_channels.g_L * (V - self.ion_channels.E_L)

        # Voltage dynamics
        dV = -(I_Na + I_K + I_L) / self.tau

        return dV, dm, dh, dn

    def calcium_dynamics(self, V: torch.Tensor, Ca: torch.Tensor) -> torch.Tensor:
        """Enhanced calcium dynamics simulation"""
        # Voltage-dependent calcium influx
        influx = torch.sigmoid((V - self.Ca_threshold) / self.Ca_slope) * \
                (self.Ca_max - Ca)
        
        # Active calcium pumping
        pump_rate = self.Ca_pump_max * Ca / (Ca + self.Ca_pump_Kd)
        
        # Passive decay
        passive_efflux = Ca * (self.dt / self.Ca_tau)
        
        # Total calcium change
        dCa = influx - pump_rate - passive_efflux
        
        return torch.clamp(Ca + dCa, 0, self.Ca_max)

    def energy_dynamics(self, ATP: torch.Tensor, activity: torch.Tensor, 
                       spike_count: torch.Tensor) -> torch.Tensor:
        """Enhanced energy dynamics simulation"""
        # ATP synthesis (dependent on glucose and oxygen)
        synthesis = self.ATP_synthesis_rate * \
                   (self.glucose_concentration / (self.glucose_concentration + 1.0)) * \
                   (self.oxygen_concentration / (self.oxygen_concentration + 100.0))
        
        # ATP consumption from neural activity
        base_consumption = activity.abs().mean() * self.dt
        spike_consumption = spike_count * self.ATP_usage_per_spike
        
        # Total ATP change
        dATP = synthesis - base_consumption - spike_consumption
        
        return torch.clamp(ATP + dATP, self.ATP_min, self.ATP_max)

    def neurotransmitter_dynamics(self, V: torch.Tensor, NT: Dict[str, torch.Tensor], 
                                spike: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced neurotransmitter dynamics simulation"""
        new_NT = {}
        for nt_type, current_level in NT.items():
            params = self.NT_types[nt_type]
            
            # Voltage and spike-dependent release
            release = torch.sigmoid((V - self.theta) / 10) * \
                     (params['max'] - current_level) * spike
            
            # Reuptake and degradation
            clearance = current_level * (self.dt / params['tau'])
            
            # Update neurotransmitter level
            new_level = current_level + release - clearance
            new_NT[nt_type] = torch.clamp(new_level, 0, params['max'])
        
        return new_NT

    def stdp_update(self, pre_spike: torch.Tensor, post_spike: torch.Tensor, 
                   weights: torch.Tensor) -> torch.Tensor:
        """Spike-Timing-Dependent Plasticity update"""
        dt = torch.arange(-self.stdp_window, self.stdp_window, self.dt)
        
        # STDP curve
        stdp_curve = torch.where(
            dt >= 0,
            self.stdp_A_plus * torch.exp(-dt / self.stdp_tau_plus),
            self.stdp_A_minus * torch.exp(dt / self.stdp_tau_minus)
        )
        
        # Compute weight updates
        pre_times = torch.where(pre_spike)[0].float()
        post_times = torch.where(post_spike)[0].float()
        
        dw = torch.zeros_like(weights)
        for pre_t in pre_times:
            for post_t in post_times:
                t_diff = post_t - pre_t
                if abs(t_diff) <= self.stdp_window:
                    idx = int((t_diff + self.stdp_window) / self.dt)
                    dw += stdp_curve[idx]
        
        # Apply weight update with constraints
        new_weights = weights + self.eta * dw
        return torch.clamp(new_weights, 0, 1)

    def homeostatic_plasticity(self, firing_rates: torch.Tensor, 
                             weights: torch.Tensor) -> torch.Tensor:
        """Homeostatic plasticity for maintaining target firing rates"""
        # Compute rate deviation from target
        rate_deviation = firing_rates - self.target_firing_rate
        
        # Update thresholds and weights
        dw = -self.homeostatic_learning_rate * rate_deviation.unsqueeze(1) * weights
        
        # Apply weight update with constraints
        new_weights = weights + dw
        return torch.clamp(new_weights, 0, 1)

def forward(self, sensory_input: torch.Tensor, 
                state: Optional[Tuple] = None) -> Tuple[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass with enhanced biological realism
        
        Args:
            sensory_input: Tensor of shape (batch_size, 6) containing sensory inputs
            state: Optional tuple containing previous network state
            
        Returns:
            outputs: Dictionary containing network outputs and state variables
            new_state: Tuple containing updated network state
        """
        batch_size = sensory_input.shape[0]
        
        # Initialize or unpack state
        if state is None:
            V = torch.full((batch_size, 1024), self.Vrest, device=sensory_input.device)
            m = torch.zeros_like(V)  # Na activation
            h = torch.ones_like(V)   # Na inactivation
            n = torch.zeros_like(V)  # K activation
            Ca = torch.full_like(V, self.Ca_0)
            ATP = torch.full_like(V, self.ATP_0)
            NT = {nt: torch.full_like(V, params['baseline']) 
                  for nt, params in self.NT_types.items()}
            spike_count = torch.zeros_like(V)
        else:
            V, m, h, n, Ca, ATP, NT, spike_count = state

        # Process sensory inputs through hierarchical layers
        x = sensory_input
        
        # Sensory processing
        x_visual = F.relu(self.sensory_layer(x[:, 0].unsqueeze(1)))
        x_auditory = F.relu(self.sensory_layer(x[:, 1].unsqueeze(1)))
        x_tactile = F.relu(self.sensory_layer(x[:, 2].unsqueeze(1)))
        x_olfactory = F.relu(self.sensory_layer(x[:, 3].unsqueeze(1)))
        x_gustatory = F.relu(self.sensory_layer(x[:, 4].unsqueeze(1)))
        x_proprioceptive = F.relu(self.sensory_layer(x[:, 5].unsqueeze(1)))
        
        # Combine sensory inputs
        x = torch.cat([x_visual, x_auditory, x_tactile, 
                      x_olfactory, x_gustatory, x_proprioceptive], dim=1)
        
        # Primary processing
        x = F.relu(self.primary_layer(x))
        
        # Association processing
        x = F.relu(self.association_layer(x))
        
        # Higher-order processing
        x = F.relu(self.higher_layer(x))
        
        # Parallel pathways
        x_limbic = F.relu(self.limbic_layer(x))
        x_basal = F.relu(self.basal_layer(x))
        x_cerebellum = F.relu(self.cerebellum_layer(x))
        
        # Sequential processing
        x_memory = F.relu(self.memory_layer(x))
        x_executive = F.relu(self.executive_layer(x_memory))
        
        # Biophysical dynamics
        dV, dm, dh, dn = self.hodgkin_huxley_dynamics(V, m, h, n)
        
        # Update membrane potential and gate variables
        V = V + dV * self.dt
        m = m + dm * self.dt
        h = h + dh * self.dt
        n = n + dn * self.dt
        
        # Detect spikes
        spike = (V >= self.theta).float()
        spike_count = spike_count + spike
        
        # Reset after spikes
        V = torch.where(spike == 1, torch.tensor(self.Vrest), V)
        
        # Update calcium concentration
        Ca = self.calcium_dynamics(V, Ca)
        
        # Update ATP levels
        ATP = self.energy_dynamics(ATP, x, spike_count)
        
        # Update neurotransmitter levels
        NT = self.neurotransmitter_dynamics(V, NT, spike)
        
        # Generate outputs
        motor_output = self.motor_output(x_executive)
        autonomic_output = self.autonomic_output(x_limbic)
        cognitive_output = self.cognitive_output(x_executive)
        
        # Update weights based on activity
        if self.training:
            # Apply STDP
            self.W_visual = self.stdp_update(x_visual, spike, self.W_visual)
            self.W_auditory = self.stdp_update(x_auditory, spike, self.W_auditory)
            
            # Apply homeostatic plasticity
            firing_rates = spike_count / (self.dt * len(self.spike_history))
            self.W_visual = self.homeostatic_plasticity(firing_rates, self.W_visual)
            self.W_auditory = self.homeostatic_plasticity(firing_rates, self.W_auditory)
        
        # Store spike history
        self.spike_history.append(spike)
        if len(self.spike_history) > self.max_history_length:
            self.spike_history.pop(0)
        
        # Package outputs
        outputs = {
            'motor': motor_output,
            'autonomic': autonomic_output,
            'cognitive': cognitive_output,
            'membrane_potential': V,
            'calcium': Ca,
            'ATP': ATP,
            'neurotransmitters': NT,
            'spike_count': spike_count
        }
        
        # Package new state
        new_state = (V, m, h, n, Ca, ATP, NT, spike_count)
        
        return outputs, new_state
