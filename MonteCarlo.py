import numpy as np
import matplotlib.pyplot as plt

# Parameter
mu = 0.075         # Erwartete Jahresrendite (7.5% - historischer Durchschnitt)

# Heston-Modell Parameter
v0 = 0.0289        # Initiale Varianz (entspricht 17% Volatilität)
kappa = 2.0        # Mean-Reversion-Geschwindigkeit der Volatilität
theta = 0.0256     # Langfristige Varianz (entspricht 16% Volatilität)
sigma_v = 0.3      # Volatilität der Varianz (Vol-of-Vol)
rho = -0.7         # Korrelation zwischen Preis- und Volatilitäts-Schocks

# Jump-Diffusion Parameter (Merton-Modell) - angepasst für realistische Crashs
lambda_jump = 0.11      # Sprungfrequenz (~1 Crash alle 9 Jahre: 1/0.11 ≈ 9.1 Jahre)
mu_jump = -0.15         # Durchschnittliche Sprunggrößer (-15% bei Crashs)
sigma_jump = 0.08       # Volatilität der Sprünge (8% - heterogenere Crashes)

# Regime-Switching Parameter (basierend auf empirischen Studien)
regimes = {
    'bull': {'mu': 0.15, 'vol_multiplier': 0.8, 'prob': 0.60},     # Bull: 15% Rendite, niedrigere Vol
    'normal': {'mu': 0.08, 'vol_multiplier': 1.0, 'prob': 0.30},   # Normal: 8% Rendite, normale Vol
    'bear': {'mu': -0.15, 'vol_multiplier': 1.8, 'prob': 0.10}     # Bear: -15% Rendite, hohe Vol
}

# Übergangsmatrix - kalibriert für realistische Steady-State-Verteilung (~60% Bull, 30% Normal, 10% Bear)
# Regime-Dauern: Bull ~18 Monate, Normal ~8 Monate, Bear ~6 Monate
# Monatliche Übergangswahrscheinlichkeiten
transition_matrix = np.array([
    [17/18, 1/20, 1/180],          # Von Bull: 94.44% bleiben (Ø 18 Monate), 5% → Normal, 0.56% → Bear
    [2/20, 7/8, 1/40],             # Von Normal: 10% → Bull, 87.5% bleiben (Ø 8 Monate), 2.5% → Bear  
    [2/15, 7/150, 5/6]             # Von Bear: 13.33% → Bull, 4.67% → Normal, 83.33% bleiben (Ø 6 Monate)
])

# Normalisierung sicherstellen (für Rundungsfehler)
for i in range(transition_matrix.shape[0]):
    transition_matrix[i, :] = transition_matrix[i, :] / transition_matrix[i, :].sum()

# Berechne langfristige Regime-Wahrscheinlichkeiten (Steady State)
# Löse π = π * P, wo π der Eigenvektor zum Eigenwert 1 ist
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
steady_state_index = np.argmax(np.real(eigenvalues))
steady_state = np.real(eigenvectors[:, steady_state_index])
steady_state = steady_state / steady_state.sum()  # Normalisieren
print(f"📊 Empirische Regime-Verteilung (Steady State):")
print(f"   Bull Market: {steady_state[0]:.1%} (Ziel: 55-65%)")
print(f"   Normal Market: {steady_state[1]:.1%} (Ziel: 25-35%)")  
print(f"   Bear Market: {steady_state[2]:.1%} (Ziel: 5-15%)")
print(f"   Durchschnittliche Regime-Dauern: Bull={1/(1-transition_matrix[0,0]):.1f}M, Normal={1/(1-transition_matrix[1,1]):.1f}M, Bear={1/(1-transition_matrix[2,2]):.1f}M")

# Inflations-Parameter
inflation_mean = 0.022      # 2.2% durchschnittliche Inflation (EZB-Ziel: ~2%)
inflation_vol = 0.004       # Inflation-Volatilität (0.4% - stabile Geldpolitik)
inflation_persistence = 0.85 # Autokorrelation der Inflation

# Stochastische Zinsen (Vasicek-Modell)
r0 = 0.025              # Aktueller risikofreier Zinssatz (2.5%)
kappa_r = 0.8           # Mean-Reversion-Geschwindigkeit Zinsen
theta_r = 0.03          # Langfristiger Zinssatz (3%)
sigma_r = 0.012         # Zinssatz-Volatilität (1.2%)
rho_rv = -0.3           # Korrelation zwischen Zinsen und Volatilität

# Transaction Costs & Steuern
transaction_cost_rate = 0.001    # 0.1% pro Transaktion (ETF-Spread)
management_fee_rate = 0.002      # 0.2% jährliche TER (Total Expense Ratio) für ETFs
tax_rate_dividends = 0.264       # 26.375% Abgeltungsteuer auf Dividenden
tax_rate_gains = 0.264           # 26.375% auf realisierte Gewinne
freistellungsauftrag = 1000      # 1000€ jährlicher Freibetrag

# ETF Portfolio Rebalancing (3-ETF-System) - Tax-Optimized
rebalance_frequency = 24         # Rebalancing-Intervall in Monaten (24 = alle 2 Jahre)
etf1_weight = 0.5               # 50% ETF 1 (z.B. World Aktien)
etf2_weight = 0.3               # 30% ETF 2 (z.B. Emerging Markets)
etf3_weight = 0.2               # 20% ETF 3 (z.B. Anleihen/REITs)
rebalance_threshold = 0.15       # 15% Abweichung löst Rebalancing aus (statt 10%)
cash_flow_rebalancing = True     # Nutze Sparraten für passives Rebalancing
thesaurierend_etf_factor = 0.7   # 70% der Dividenden werden thesauriert (steuerfrei)

# Fat-Tail Parameter (t-Verteilung)
df_shocks = 6               # Freiheitsgrade für t-verteilte Schocks (fat tails)

# === BLACK SWAN EVENT PARAMETERS ===
# Hyperinflation Events (1970s/Weimar-Style) - REALISTISCH für entwickelte Märkte
hyperinflation_prob = 1/150         # Etwa alle 150 Jahre (OECD nach 1945: quasi inexistent, 1-3% über 30J)
hyperinflation_rate = 0.12          # 12% Inflation während Hyperinflation-Jahren
hyperinflation_duration_years = 3   # Typische Dauer: 2-4 Jahre
hyperinflation_vol = 0.03           # Zusätzliche Inflation-Volatilität

# Strukturbruch/Systemkrise Events - REALISTISCH: alle 10-15 Jahre eine große Krise
structural_crisis_prob = 1/12       # Etwa alle 12 Jahre (80-90% Chance über 30 Jahre, 2-4 Krisen)
structural_crisis_drawdown = 0.50   # 50% Markt-Drawdown über 2-3 Jahre
structural_crisis_duration_months = 24  # 2 Jahre anhaltende Krise
structural_crisis_recovery_years = 5    # Langsame Erholung über 5 Jahre

# Politische/Steuerliche Schocks - REALISTISCH: 40-60% über 30 Jahre
tax_shock_prob = 1/30              # Alle 30 Jahre große Steuerreformen (50% über 30 Jahre)
tax_shock_new_rate = 0.45          # Kapitalertragssteuer steigt auf 45%
tax_shock_wealth_tax = 0.01        # 1% Vermögenssteuer ab 500k€

# Persönliche Liquidationskrisen - REALISTISCH: 25-40% über 30 Jahre
personal_crisis_prob_per_year = 0.015  # 1.5% Chance pro Jahr (→ ~35% über 30J, statistisch korrekt)
liquidation_percentage = 0.30          # 30% des Portfolios muss liquidiert werden
crisis_duration_months = 6             # Krise dauert 6 Monate

num_sim = 50000     # Anzahl Simulationen (reduziert für Test)
jahre = 26         # Anlagehorizont in Jahren
startwert = 12000    # Startkurs in Euro
monatliche_sparrate = 500  # Monatliche Sparrate in Euro

# --- Erweiterte Heston-Simulation mit Jump-Diffusion, Regime-Switching und Inflation ---
np.random.seed(42)  # Für reproduzierbare Ergebnisse

def sample_regime(current_regime, transition_matrix):
    """Wählt nächstes Regime basierend auf Übergangswahrscheinlichkeiten"""
    probs = transition_matrix[current_regime]
    return np.random.choice(len(probs), p=probs)

def simulate_stochastic_rates(r0, kappa_r, theta_r, sigma_r, T, steps, num_sim):
    """Simuliert stochastische Zinssätze mit Vasicek-Modell"""
    dt = T / steps
    rates = np.zeros((num_sim, steps + 1))
    rates[:, 0] = r0
    
    for t in range(steps):
        dW_r = np.random.normal(0, np.sqrt(dt), num_sim)
        rates[:, t+1] = (rates[:, t] + 
                        kappa_r * (theta_r - rates[:, t]) * dt + 
                        sigma_r * dW_r)
        # Zinssätze können nicht stark negativ werden
        rates[:, t+1] = np.maximum(rates[:, t+1], -0.01)
    
    return rates

def generate_t_distributed_shocks(df, size):
    """
    Generiert t-verteilte Schocks für fat tails.
    Hinweis: Die Normalisierung auf Einheitsvarianz ist nur für df > 2 gültig.
    """
    # t-verteilte Zufallszahlen, normalisiert auf Einheitsvarianz
    shocks = np.random.standard_t(df, size)
    # Normalisierung damit Varianz = 1
    if df > 2:
        shocks = shocks / np.sqrt(df / (df - 2))
    return shocks

def calculate_transaction_costs(portfolio_value_before, portfolio_value_after, cost_rate):
    """Berechnet Transaktionskosten basierend auf Handelswert"""
    trade_value = abs(portfolio_value_after - portfolio_value_before)
    return trade_value * cost_rate

def apply_taxes_and_fees(portfolio_value, monthly_gain_euro, management_fee_rate, 
                        tax_rate, freibetrag_remaining, dt):
    """Wendet Steuern und Gebühren an"""
    # Management Fee (anteilig)
    management_fee = portfolio_value * management_fee_rate * dt
    
    # Steuern nur auf positive Gewinne in Euro
    if monthly_gain_euro > 0:
        monthly_freibetrag = freibetrag_remaining / 12  # Monatlicher Freibetrag
        taxable_gain_euro = max(0, monthly_gain_euro - monthly_freibetrag)
        taxes = taxable_gain_euro * tax_rate
        freibetrag_used = min(monthly_gain_euro, monthly_freibetrag)
        freibetrag_remaining = max(0, freibetrag_remaining - freibetrag_used)
    else:
        taxes = 0
        
    total_costs = management_fee + taxes
    return total_costs, freibetrag_remaining

def simulate_black_swan_events(T, steps, num_sim):
    """Simuliert verschiedene Black Swan Events für realistische Tail-Risiken"""
    dt = T / steps
    
    # Event-Tracking Arrays
    hyperinflation_active = np.zeros((num_sim, steps + 1), dtype=bool)
    structural_crisis_active = np.zeros((num_sim, steps + 1), dtype=bool)
    tax_shock_active = np.zeros((num_sim, steps + 1), dtype=bool)
    personal_crisis_active = np.zeros((num_sim, steps + 1), dtype=bool)
    
    # Event-Counters für Statistiken
    event_counts = {
        'hyperinflation': 0,
        'structural_crisis': 0, 
        'tax_shock': 0,
        'personal_crisis': 0
    }
    
    for sim in range(num_sim):
        # Hyperinflation Events (korrekte jährliche Wahrscheinlichkeit)
        hyperinflation_start = None
        # Berechne Wahrscheinlichkeit: 1 - (1 - p_per_year)^T
        hyperinflation_total_prob = 1 - (1 - hyperinflation_prob) ** T
        if np.random.random() < hyperinflation_total_prob:
            hyperinflation_start = np.random.randint(0, max(1, steps - hyperinflation_duration_years * 12))
            hyperinflation_end = min(steps, hyperinflation_start + hyperinflation_duration_years * 12)
            hyperinflation_active[sim, hyperinflation_start:hyperinflation_end] = True
            event_counts['hyperinflation'] += 1
        
        # Strukturkrisen (korrekte jährliche Wahrscheinlichkeit)
        structural_total_prob = 1 - (1 - structural_crisis_prob) ** T
        if np.random.random() < structural_total_prob:
            crisis_start = np.random.randint(0, max(1, steps - structural_crisis_duration_months))
            crisis_end = min(steps, crisis_start + structural_crisis_duration_months)
            structural_crisis_active[sim, crisis_start:crisis_end] = True
            event_counts['structural_crisis'] += 1
        
        # Steuer-Schocks (korrekte jährliche Wahrscheinlichkeit)
        tax_total_prob = 1 - (1 - tax_shock_prob) ** T
        if np.random.random() < tax_total_prob:
            tax_shock_start = np.random.randint(steps//3, steps)  # Nur in zweiter Hälfte
            tax_shock_active[sim, tax_shock_start:] = True
            event_counts['tax_shock'] += 1
        
        # Persönliche Krisen (kurze Perioden, aber häufiger)
        for year in range(int(T)):
            if np.random.random() < personal_crisis_prob_per_year:
                crisis_month = year * 12 + np.random.randint(0, 12)
                if crisis_month < steps:
                    crisis_end = min(steps, crisis_month + crisis_duration_months)
                    personal_crisis_active[sim, crisis_month:crisis_end] = True
                    event_counts['personal_crisis'] += 1
    
    return (hyperinflation_active, structural_crisis_active, 
            tax_shock_active, personal_crisis_active, event_counts)

def simulate_inflation(T, steps, num_sim):
    """Simuliert stochastische Inflation mit AR(1)-Prozess + Black Swan Events"""
    dt = T / steps
    inflation_paths = np.zeros((num_sim, steps + 1))
    inflation_paths[:, 0] = inflation_mean
    
    # Black Swan Events simulieren
    (hyperinflation_events, _, _, _, _) = simulate_black_swan_events(T, steps, num_sim)
    
    for t in range(steps):
        # Standard AR(1)-Prozess für Inflation
        innovation = np.random.normal(0, inflation_vol * np.sqrt(dt), num_sim)
        inflation_paths[:, t+1] = (
            (1 - inflation_persistence) * inflation_mean + 
            inflation_persistence * inflation_paths[:, t] + 
            innovation
        )
        
        # Hyperinflation Events anwenden
        hyperinflation_mask = hyperinflation_events[:, t+1]
        if np.any(hyperinflation_mask):
            # Während Hyperinflation: Deutlich höhere Inflation + Volatilität
            hyperinflation_shock = np.random.normal(hyperinflation_rate, hyperinflation_vol, num_sim)
            inflation_paths[hyperinflation_mask, t+1] = hyperinflation_shock[hyperinflation_mask]
        
        # Inflation kann nicht stark negativ werden (Deflationsgrenze)
        inflation_paths[:, t+1] = np.maximum(inflation_paths[:, t+1], -0.02)
    
    return inflation_paths

def heston_jump_regime_simulation(S0, v0, mu_base, kappa, theta, sigma_v, rho, T, num_sim):
    """Vollständig erweiterte 1-Jahres-Simulation mit allen Phase 1 & 2 Features"""
    dt = T / 252  # Tägliche Schritte
    steps = 252
    
    # Inflation simulieren
    inflation_paths = simulate_inflation(T, steps, num_sim)
    
    # Stochastische Zinssätze simulieren
    interest_rates = simulate_stochastic_rates(r0, kappa_r, theta_r, sigma_r, T, steps, num_sim)
    
    # t-verteilte Schocks für fat tails
    dW1_normal = np.random.normal(0, 1, (num_sim, steps))
    dW2_normal = np.random.normal(0, 1, (num_sim, steps))
    
    # Fat-tail Schocks
    t_shocks1 = generate_t_distributed_shocks(df_shocks, (num_sim, steps))
    t_shocks2 = generate_t_distributed_shocks(df_shocks, (num_sim, steps))
    
    # Korrelierte Brownsche Bewegungen mit t-Verteilung
    dW1 = t_shocks1 * np.sqrt(dt)
    dW2_indep = t_shocks2 * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2_indep
    
    # Korrelation zwischen Zinsen und Volatilität
    dW_r = rho_rv * dW2 + np.sqrt(1 - rho_rv**2) * np.random.normal(0, np.sqrt(dt), (num_sim, steps))
    
    # Arrays initialisieren
    S = np.zeros((num_sim, steps + 1))
    v = np.zeros((num_sim, steps + 1))
    regimes_path = np.zeros((num_sim, steps + 1), dtype=int)
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Initiales Regime (Bull=0, Normal=1, Bear=2)
    regime_probs = [regimes['bull']['prob'], regimes['normal']['prob'], regimes['bear']['prob']]
    regimes_path[:, 0] = np.random.choice(3, size=num_sim, p=regime_probs)
    
    # Simulation über alle Zeitschritte
    for t in range(steps):
        # Regime-Switching für jede Simulation
        for sim in range(num_sim):
            regimes_path[sim, t+1] = sample_regime(regimes_path[sim, t], transition_matrix)
        
        # Regime-abhängige Parameter
        current_regimes = regimes_path[:, t]
        mu_regime = np.zeros(num_sim)
        vol_multipliers = np.zeros(num_sim)
        
        for sim in range(num_sim):
            if current_regimes[sim] == 0:  # Bull
                mu_regime[sim] = regimes['bull']['mu']
                vol_multipliers[sim] = regimes['bull']['vol_multiplier']
            elif current_regimes[sim] == 1:  # Normal
                mu_regime[sim] = regimes['normal']['mu']
                vol_multipliers[sim] = regimes['normal']['vol_multiplier']
            else:  # Bear
                mu_regime[sim] = regimes['bear']['mu']
                vol_multipliers[sim] = regimes['bear']['vol_multiplier']
        
        # Note: Black Swan events are only applied in the long-term portfolio simulation
        
        # Inflationsanpassung der Renditen (real returns)
        inflation_rate = inflation_paths[:, t]
        mu_real = mu_regime - inflation_rate
        
        v_pos = np.maximum(v[:, t], 1e-8)  # Volatilität kann nicht negativ werden
        v_adjusted = v_pos * (vol_multipliers ** 2)  # Regime-abhängige Volatilität
        
        # Varianz-Prozess (CIR) mit Regime-Einfluss
        v[:, t+1] = v[:, t] + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos) * dW2[:, t]
        v[:, t+1] = np.maximum(v[:, t+1], 1e-8)
        
        # Jump-Komponente (Merton-Modell)
        jump_occurs = np.random.poisson(lambda_jump * dt, num_sim) > 0
        jump_sizes = np.random.normal(mu_jump, sigma_jump, num_sim)
        jump_component = jump_occurs * jump_sizes
        
        # Erweiterte Preis-Dynamik: Heston + Jumps + Regime + Inflation
        drift = mu_real - 0.5 * v_adjusted
        diffusion = np.sqrt(v_adjusted) * dW1[:, t]
        
        S[:, t+1] = S[:, t] * np.exp(drift * dt + diffusion + jump_component)
    
    # 1-Jahres-Renditen berechnen (nominal)
    renditen_nominal = (S[:, -1] - S[:, 0]) / S[:, 0]
    
    # Reale Renditen (inflationsbereinigt)
    avg_inflation = inflation_paths.mean(axis=1)
    renditen_real = renditen_nominal - avg_inflation
    
    return renditen_nominal, renditen_real, inflation_paths

# 1-Jahres-Simulation durchführen
renditen_1y_nominal, renditen_1y_real, inflation_1y = heston_jump_regime_simulation(
    startwert, v0, mu, kappa, theta, sigma_v, rho, 1.0, num_sim
)

# Verwende reale Renditen für VaR-Berechnungen
renditen_1y = renditen_1y_real

var75_1y = np.percentile(renditen_1y, 25)  # 25% VaR
var95_1y = np.percentile(renditen_1y, 5)
var99_1y = np.percentile(renditen_1y, 1)
es95_1y = renditen_1y[renditen_1y <= var95_1y].mean()

# 67%-Konfidenzintervall (±1 Standardabweichung)
conf67_lower_1y = np.percentile(renditen_1y, 16.5)
conf67_upper_1y = np.percentile(renditen_1y, 83.5)

print("--- 1 Jahr (Reale Renditen nach Inflation) ---")
print(f"Durchschnittliche Inflation: {inflation_1y.mean():.2%}")
print(f"Nominale Rendite (Durchschnitt): {renditen_1y_nominal.mean():.2%}")
print(f"Reale Rendite (Durchschnitt): {renditen_1y.mean():.2%}")
print(f"Standardabweichung (real): {renditen_1y.std():.2%}")
print(f"67%-Konfidenzintervall: {conf67_lower_1y:.2%} bis {conf67_upper_1y:.2%}")
print(f"75% VaR: {var75_1y:.2%}")
print(f"95% VaR: {var95_1y:.2%}")
print(f"99% VaR: {var99_1y:.2%}")
print(f"95% Expected Shortfall: {es95_1y:.2%}\n")

# --- Mehrjährige erweiterte Simulation ---
def heston_portfolio_extended_simulation(S0, v0, mu_base, kappa, theta, sigma_v, rho, T, steps, num_sim, sparrate):
    """3-ETF Portfolio-Simulation mit jährlichem Rebalancing + Black Swan Events"""
    dt = T / steps
    
    # Black Swan Events simulieren
    (hyperinflation_events, structural_crisis_events, 
     tax_shock_events, personal_crisis_events, event_counts) = simulate_black_swan_events(T, steps, num_sim)
    
    print(f"🦢 Black Swan Events simuliert (Realistische OECD-Wahrscheinlichkeiten):")
    print(f"   Hyperinflation-Perioden: {event_counts['hyperinflation']} von {num_sim} Simulationen")
    print(f"   Strukturkrisen: {event_counts['structural_crisis']} von {num_sim}")
    print(f"   Steuer-Schocks: {event_counts['tax_shock']} von {num_sim}")
    print(f"   Persönliche Krisen: {event_counts['personal_crisis']} von {num_sim}")
    
    # Inflation über gesamten Zeitraum simulieren (bereits mit Hyperinflation)
    inflation_paths = simulate_inflation(T, steps, num_sim)
    
    # Stochastische Zinssätze simulieren
    interest_rates = simulate_stochastic_rates(r0, kappa_r, theta_r, sigma_r, T, steps, num_sim)
    
    # t-verteilte Schocks für realistische Fat Tails
    t_shocks1 = generate_t_distributed_shocks(df_shocks, (num_sim, steps))
    t_shocks2 = generate_t_distributed_shocks(df_shocks, (num_sim, steps))
    t_shocks3 = generate_t_distributed_shocks(df_shocks, (num_sim, steps))  # Für ETF 3
    
    # Korrelierte Brownsche Bewegungen mit Fat Tails
    dW1 = t_shocks1 * np.sqrt(dt)
    dW2_indep = t_shocks2 * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2_indep
    dW3 = 0.3 * dW1 + 0.2 * dW2 + np.sqrt(1 - 0.3**2 - 0.2**2) * t_shocks3 * np.sqrt(dt)  # ETF 3 Korrelation
    
    # Zinsen-Volatilitäts-Korrelation
    dW_r = rho_rv * dW2 + np.sqrt(1 - rho_rv**2) * generate_t_distributed_shocks(df_shocks, (num_sim, steps)) * np.sqrt(dt)
    
    # 3-ETF Portfolio Arrays
    portfolio_werte = np.zeros((num_sim, steps + 1))
    portfolio_werte_real = np.zeros((num_sim, steps + 1))
    v = np.zeros((num_sim, steps + 1))
    
    # ETF Positionen (3 ETFs)
    etf1_wert = np.zeros((num_sim, steps + 1))  # 50% - World Aktien
    etf2_wert = np.zeros((num_sim, steps + 1))  # 30% - Emerging Markets
    etf3_wert = np.zeros((num_sim, steps + 1))  # 20% - Anleihen/REITs
    etf1_anteile = np.zeros((num_sim, steps + 1))
    etf2_anteile = np.zeros((num_sim, steps + 1))
    etf3_anteile = np.zeros((num_sim, steps + 1))
    
    # ETF Kurse
    etf1_kurs = np.zeros((num_sim, steps + 1))  
    etf2_kurs = np.zeros((num_sim, steps + 1))  
    etf3_kurs = np.zeros((num_sim, steps + 1))  
    
    # Kosten-Tracking
    kumulierte_kosten = np.zeros((num_sim, steps + 1))
    kumulierte_steuern = np.zeros((num_sim, steps + 1))
    kumulierte_ter = np.zeros((num_sim, steps + 1))  # Separate TER-Tracking
    freibetrag_remaining = np.full(num_sim, freistellungsauftrag)
    
    regimes_path = np.zeros((num_sim, steps + 1), dtype=int)
    
    # Rebalancing-Zähler (jährlich)
    last_rebalance = np.zeros(num_sim, dtype=int)
    
    # CASH-FLOW REBALANCING: Persistente Allokations-Gewichtungen
    # Diese werden zwischen Iterationen gespeichert und angepasst
    current_allocation_weights = np.array([etf1_weight, etf2_weight, etf3_weight])
    allocation_adjustment_active = False
    
    # Initialisierung 3-ETF Portfolio
    portfolio_werte[:, 0] = S0
    portfolio_werte_real[:, 0] = S0
    v[:, 0] = v0
    
    # Initiale ETF-Allokation (50%/30%/20%)
    etf1_wert[:, 0] = S0 * etf1_weight
    etf2_wert[:, 0] = S0 * etf2_weight
    etf3_wert[:, 0] = S0 * etf3_weight
    etf1_kurs[:, 0] = 1.0
    etf2_kurs[:, 0] = 1.0
    etf3_kurs[:, 0] = 1.0
    etf1_anteile[:, 0] = etf1_wert[:, 0] / etf1_kurs[:, 0]
    etf2_anteile[:, 0] = etf2_wert[:, 0] / etf2_kurs[:, 0]
    etf3_anteile[:, 0] = etf3_wert[:, 0] / etf3_kurs[:, 0]
    
    # Initiales Regime
    regime_probs = [regimes['bull']['prob'], regimes['normal']['prob'], regimes['bear']['prob']]
    regimes_path[:, 0] = np.random.choice(3, size=num_sim, p=regime_probs)
    
    # Simulation über alle Zeitschritte
    for t in range(steps):
        # Regime-Switching
        for sim in range(num_sim):
            regimes_path[sim, t+1] = sample_regime(regimes_path[sim, t], transition_matrix)
        
        # Regime-Parameter
        current_regimes = regimes_path[:, t]
        mu_regime = np.zeros(num_sim)
        vol_multipliers = np.zeros(num_sim)
        
        for sim in range(num_sim):
            if current_regimes[sim] == 0:  # Bull
                mu_regime[sim] = regimes['bull']['mu']
                vol_multipliers[sim] = regimes['bull']['vol_multiplier']
            elif current_regimes[sim] == 1:  # Normal
                mu_regime[sim] = regimes['normal']['mu']
                vol_multipliers[sim] = regimes['normal']['vol_multiplier']
            else:  # Bear
                mu_regime[sim] = regimes['bear']['mu']
                vol_multipliers[sim] = regimes['bear']['vol_multiplier']
        
        # Inflation
        inflation_rate = inflation_paths[:, t]
        mu_real = mu_regime - inflation_rate
        
        v_pos = np.maximum(v[:, t], 1e-8)
        v_adjusted = v_pos * (vol_multipliers ** 2)
        
        # CIR-Prozess für Varianz
        v[:, t+1] = v[:, t] + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos) * dW2[:, t]
        v[:, t+1] = np.maximum(v[:, t+1], 1e-8)
        
        # Jump-Diffusion
        jump_occurs = np.random.poisson(lambda_jump * dt, num_sim) > 0
        jump_sizes = np.random.normal(mu_jump, sigma_jump, num_sim)
        jump_component = jump_occurs * jump_sizes
        
        # ETF 1 Kursentwicklung (World Aktien - Heston-Modell)
        drift_etf1 = mu_real - 0.5 * v_adjusted
        diffusion_etf1 = np.sqrt(v_adjusted) * dW1[:, t]
        etf1_kurs[:, t+1] = etf1_kurs[:, t] * np.exp(drift_etf1 * dt + diffusion_etf1 + jump_component)
        
        # ETF 2 Kursentwicklung (Emerging Markets - höhere Volatilität)
        drift_etf2 = mu_real * 1.1 - 0.5 * v_adjusted * 1.3  # Leicht höhere Rendite, höhere Vol
        diffusion_etf2 = np.sqrt(v_adjusted * 1.3) * dW2[:, t]
        etf2_kurs[:, t+1] = etf2_kurs[:, t] * np.exp(drift_etf2 * dt + diffusion_etf2 + jump_component * 1.2)
        
        # ETF 3 Kursentwicklung (Anleihen/REITs - niedriger Volatilität, Zinskorrelation)
        bond_return = interest_rates[:, t] - (interest_rates[:, t+1] - interest_rates[:, t]) * 3  # Duration ~3 Jahre
        drift_etf3 = bond_return * 0.7 + mu_real * 0.3  # Mix aus Zins- und Aktienrendite
        etf3_volatility = 0.06  # 6% Volatilität
        etf3_kurs[:, t+1] = etf3_kurs[:, t] * np.exp(
            (drift_etf3 - 0.5 * etf3_volatility**2) * dt + 
            etf3_volatility * dW3[:, t]
        )
        
        # Neue Investitionen mit Cash-Flow-Rebalancing Gewichtungen
        # Nutze persistente Allokation aus vorherigem Monat
        neue_etf1_invest = sparrate * current_allocation_weights[0]
        neue_etf2_invest = sparrate * current_allocation_weights[1]  
        neue_etf3_invest = sparrate * current_allocation_weights[2]
        
        neue_etf1_anteile = neue_etf1_invest / etf1_kurs[:, t+1]
        neue_etf2_anteile = neue_etf2_invest / etf2_kurs[:, t+1]
        neue_etf3_anteile = neue_etf3_invest / etf3_kurs[:, t+1]
        
        etf1_anteile[:, t+1] = etf1_anteile[:, t] + neue_etf1_anteile
        etf2_anteile[:, t+1] = etf2_anteile[:, t] + neue_etf2_anteile
        etf3_anteile[:, t+1] = etf3_anteile[:, t] + neue_etf3_anteile
        
        # Portfolio-Werte vor Rebalancing und Kosten
        etf1_wert[:, t+1] = etf1_anteile[:, t+1] * etf1_kurs[:, t+1]
        etf2_wert[:, t+1] = etf2_anteile[:, t+1] * etf2_kurs[:, t+1]
        etf3_wert[:, t+1] = etf3_anteile[:, t+1] * etf3_kurs[:, t+1]
        portfolio_werte_gross = etf1_wert[:, t+1] + etf2_wert[:, t+1] + etf3_wert[:, t+1]
        
        # CASH-FLOW REBALANCING: Berechne Anpassungen für NÄCHSTEN Monat
        if cash_flow_rebalancing and t < steps_monatlich - 1:  # Not for last month
            total_weight1 = 0
            total_weight2 = 0
            total_weight3 = 0
            valid_sims = 0
            
            for sim in range(num_sim):
                if portfolio_werte_gross[sim] > 0:
                    current_etf1_weight = etf1_wert[sim, t+1] / portfolio_werte_gross[sim]
                    current_etf2_weight = etf2_wert[sim, t+1] / portfolio_werte_gross[sim]
                    current_etf3_weight = etf3_wert[sim, t+1] / portfolio_werte_gross[sim]
                    
                    total_weight1 += current_etf1_weight
                    total_weight2 += current_etf2_weight
                    total_weight3 += current_etf3_weight
                    valid_sims += 1
            
            if valid_sims > 0:
                avg_weight1 = total_weight1 / valid_sims
                avg_weight2 = total_weight2 / valid_sims
                avg_weight3 = total_weight3 / valid_sims
                
                # Calculate deviations from target
                weight_diff1 = etf1_weight - avg_weight1
                weight_diff2 = etf2_weight - avg_weight2
                weight_diff3 = etf3_weight - avg_weight3
                
                # Calculate adjustments for NEXT month's allocation
                boost_factor = 0.4  # 40% of deviation corrected through cash flow
                adj1 = etf1_weight
                adj2 = etf2_weight
                adj3 = etf3_weight
                
                # Apply adjustments only if significant deviation (> 3%)
                if abs(weight_diff1) > 0.03:
                    adj1 = etf1_weight + weight_diff1 * boost_factor
                if abs(weight_diff2) > 0.03:
                    adj2 = etf2_weight + weight_diff2 * boost_factor
                if abs(weight_diff3) > 0.03:
                    adj3 = etf3_weight + weight_diff3 * boost_factor
                
                # Normalize so sum = 1
                total_adj = adj1 + adj2 + adj3
                if total_adj > 0:
                    adj1 = adj1 / total_adj
                    adj2 = adj2 / total_adj
                    adj3 = adj3 / total_adj
                    
                    # SPEICHERE Anpassungen für nächsten Monat (t+1)
                    current_allocation_weights[0] = adj1
                    current_allocation_weights[1] = adj2
                    current_allocation_weights[2] = adj3
                    allocation_adjustment_active = True
        
        # Reset zu Standard-Allokation falls keine Anpassung aktiv
        elif not allocation_adjustment_active:
            current_allocation_weights[0] = etf1_weight
            current_allocation_weights[1] = etf2_weight
            current_allocation_weights[2] = etf3_weight
        
        # Rebalancing (alle 24 Monate für Tax-Effizienz)
        rebalance_now = (t % rebalance_frequency == 0) & (t > 0)
        
        for sim in range(num_sim):
            if rebalance_now:
                current_etf1_weight = etf1_wert[sim, t+1] / portfolio_werte_gross[sim]
                current_etf2_weight = etf2_wert[sim, t+1] / portfolio_werte_gross[sim] 
                current_etf3_weight = etf3_wert[sim, t+1] / portfolio_werte_gross[sim]
                
                weight_deviation = (abs(current_etf1_weight - etf1_weight) + 
                                  abs(current_etf2_weight - etf2_weight) +
                                  abs(current_etf3_weight - etf3_weight))
                
                if weight_deviation > rebalance_threshold:
                    # STEUEROPTIMIERTES REBALANCING:
                    # 1) Minimiere Verkäufe durch bevorzugten Kauf untergewichteter ETFs
                    # 2) Nutze verfügbare Verluste für Steuerverlustverrechnung
                    
                    target_etf1_wert = portfolio_werte_gross[sim] * etf1_weight
                    target_etf2_wert = portfolio_werte_gross[sim] * etf2_weight
                    target_etf3_wert = portfolio_werte_gross[sim] * etf3_weight
                    
                    # Reduzierte Transaction Costs durch weniger Turnover
                    # Cash-Flow-Rebalancing reduziert notwendige Trades
                    turnover_reduction = 0.6 if cash_flow_rebalancing else 1.0
                    
                    trade_volume = (abs(target_etf1_wert - etf1_wert[sim, t+1]) +
                                   abs(target_etf2_wert - etf2_wert[sim, t+1]) +
                                   abs(target_etf3_wert - etf3_wert[sim, t+1])) * turnover_reduction
                    
                    # Niedrigere Transaktionskosten durch seltenereres, größeres Rebalancing
                    effective_cost_rate = transaction_cost_rate * 0.8  # 20% Reduktion durch Bulk-Trades
                    transaction_costs = trade_volume * effective_cost_rate
                    kumulierte_kosten[sim, t+1] = kumulierte_kosten[sim, t] + transaction_costs
                    
                    # Neue Anteile nach Rebalancing
                    etf1_anteile[sim, t+1] = target_etf1_wert / etf1_kurs[sim, t+1]
                    etf2_anteile[sim, t+1] = target_etf2_wert / etf2_kurs[sim, t+1]
                    etf3_anteile[sim, t+1] = target_etf3_wert / etf3_kurs[sim, t+1]
                    
                    # Portfolio-Wert nach Transaction Costs
                    portfolio_werte_gross[sim] -= transaction_costs
        
        # Management Fees und realistische Steuern (nur Dividenden + realisierte Gewinne)
        for sim in range(num_sim):
            # Management Fee (TER) - monatlich
            management_fee = portfolio_werte_gross[sim] * management_fee_rate * dt
            
            # Steuern nur bei besonderen Ereignissen
            taxes_this_month = 0
            
            # 1) Dividenden-Steuern (mit thesaurierenden ETF-Vorteilen)
            if t % 12 == 11:  # Ende des Jahres - Dividenden treatment
                dividend_yield = 0.02  # 2% jährliche Dividende
                annual_dividends = portfolio_werte_gross[sim] * dividend_yield
                
                # THESAURIERENDE ETFs: Großteil der Dividenden steuerfrei reinvestiert
                taxable_dividend_portion = annual_dividends * (1 - thesaurierend_etf_factor)
                thesauriert_dividends = annual_dividends * thesaurierend_etf_factor
                
                # Black Swan: Steuer-Schocks anwenden
                current_tax_rate = tax_rate_dividends
                if tax_shock_events[sim, t]:
                    current_tax_rate = tax_shock_new_rate  # Drastische Steuererhöhung
                
                # Freibetrag nur auf steuerpflichtige Dividenden anwenden
                taxable_dividends = max(0, taxable_dividend_portion - freibetrag_remaining[sim])
                taxes_this_month = taxable_dividends * current_tax_rate
                freibetrag_remaining[sim] = max(0, freibetrag_remaining[sim] - taxable_dividend_portion)
                
                # Thesaurierte Dividenden erhöhen Portfolio-Wert steuerfrei
                # (Dies wird in der Kursentwicklung bereits berücksichtigt)
            
            # 2) Realisierte Gewinne bei Rebalancing (deutlich reduziert)
            if rebalance_now and t > 12:  # Nach dem ersten Jahr
                # REALISTISCHERE ANNAHME: Weniger realisierte Gewinne durch:
                # - Seltenereres Rebalancing (alle 2 Jahre statt jährlich)
                # - Cash-Flow-Rebalancing reduziert Verkäufe
                # - Nur bei großen Abweichungen (15% statt 10%)
                estimated_total_gain = max(0, portfolio_werte_gross[sim] - startwert - sparrate * (t + 1))
                
                # Reduzierte realisierte Gewinne durch Tax-Optimization
                base_realized_fraction = 0.03  # Nur 3% statt 10% (Cash-Flow-Rebalancing)
                
                # Weitere Reduktion wenn Cash-Flow-Rebalancing aktiv ist
                if cash_flow_rebalancing:
                    realized_gain_fraction = base_realized_fraction * 0.5  # Nochmals halbiert
                else:
                    realized_gain_fraction = base_realized_fraction
                
                realized_gains = estimated_total_gain * realized_gain_fraction
                
                # Black Swan: Steuer-Schocks
                current_tax_rate = tax_rate_gains
                if tax_shock_events[sim, t]:
                    current_tax_rate = tax_shock_new_rate
                
                # Steuern auf realisierte Gewinne
                taxable_gains = max(0, realized_gains - freibetrag_remaining[sim])
                rebalance_taxes = taxable_gains * current_tax_rate
                taxes_this_month += rebalance_taxes
                freibetrag_remaining[sim] = max(0, freibetrag_remaining[sim] - realized_gains)
            
            # 3) Vermögenssteuer bei Black Swan Tax Events
            if tax_shock_events[sim, t] and portfolio_werte_gross[sim] > 500000:
                wealth_tax = (portfolio_werte_gross[sim] - 500000) * tax_shock_wealth_tax * dt
                taxes_this_month += wealth_tax
            
            # Kosten zusammenfassen
            total_costs = management_fee + taxes_this_month
            
            kumulierte_kosten[sim, t+1] = kumulierte_kosten[sim, t] + total_costs
            kumulierte_steuern[sim, t+1] = kumulierte_steuern[sim, t] + taxes_this_month
            kumulierte_ter[sim, t+1] = kumulierte_ter[sim, t] + management_fee
        
        # Black Swan: Persönliche Liquidationskrisen
        for sim in range(num_sim):
            if personal_crisis_events[sim, t]:
                # Notfall-Liquidation von 30% des Portfolios zu ungünstigen Konditionen
                liquidation_amount = portfolio_werte_gross[sim] * liquidation_percentage
                liquidation_penalty = liquidation_amount * 0.05  # 5% Penalty (Bid-Ask + Timing)
                
                portfolio_werte_gross[sim] -= liquidation_amount
                kumulierte_kosten[sim, t+1] += liquidation_penalty
                
                # Proportionale Reduktion aller ETF-Anteile
                etf1_anteile[sim, t+1] *= (1 - liquidation_percentage)
                etf2_anteile[sim, t+1] *= (1 - liquidation_percentage)
                etf3_anteile[sim, t+1] *= (1 - liquidation_percentage)
        
        # Finale Portfolio-Werte nach allen Kosten
        portfolio_werte[:, t+1] = portfolio_werte_gross - (kumulierte_kosten[:, t+1] - kumulierte_kosten[:, t])
        
        # Reale Werte (inflationsbereinigt, geometrisch kumuliert)
        kum_inflation = np.prod(1 + inflation_paths[:, :t+2] * dt, axis=1) - 1
        portfolio_werte_real[:, t+1] = portfolio_werte[:, t+1] / (1 + kum_inflation)
    
    return (portfolio_werte, portfolio_werte_real, v, etf1_kurs, etf2_kurs, etf3_kurs,
            inflation_paths, interest_rates, kumulierte_kosten, kumulierte_steuern, kumulierte_ter,
            hyperinflation_events, structural_crisis_events, tax_shock_events, personal_crisis_events, event_counts)

# Zeitdiskretisierung auf Monatsbasis
T = jahre
steps_monatlich = jahre * 12

# 3-ETF Portfolio Simulation mit Black Swan Events durchführen
(portfolio_werte, portfolio_werte_real, volatilitat_pfade, etf1_kurse, etf2_kurse, etf3_kurse,
 inflation_pfade, zinssatz_pfade, kosten_pfade, steuer_pfade, ter_pfade,
 hyperinflation_events, structural_crisis_events, tax_shock_events, personal_crisis_events, event_counts) = heston_portfolio_extended_simulation(
    startwert, v0, mu, kappa, theta, sigma_v, rho, T, steps_monatlich, num_sim, monatliche_sparrate
)

# Jährliche Werte für Grafiken (nominale Werte)
jahres_indizes = np.arange(11, steps_monatlich, 12)
kurse = portfolio_werte[:, jahres_indizes]

# Endwerte (nominal und real)
endwerte_nominal = portfolio_werte[:, -1]
endwerte_real = portfolio_werte_real[:, -1]

# Verwende reale Endwerte für Hauptanalyse
endwerte = endwerte_real

# Berechne Gesamteinzahlungen (nominal)
gesamteinzahlungen_nominal = startwert + (monatliche_sparrate * steps_monatlich)

# Inflationsanpassung der Einzahlungen für faire Vergleiche
avg_inflation = inflation_pfade.mean()
gesamteinzahlungen_real = gesamteinzahlungen_nominal / (1 + avg_inflation * T)

gewinn_verlust_real = endwerte - gesamteinzahlungen_real

# Risikokennzahlen für reale Endwerte
var75_T = np.percentile(endwerte, 25)
var95_T = np.percentile(endwerte, 5)
var99_T = np.percentile(endwerte, 1)

# Konfidenzintervalle für Endwerte
conf67_lower_T = np.percentile(endwerte, 16.5)   # 67% (±1σ)
conf67_upper_T = np.percentile(endwerte, 83.5)
conf90_lower_T = np.percentile(endwerte, 5.0)    # 90% Konfidenzintervall
conf90_upper_T = np.percentile(endwerte, 95.0)
conf95_lower_T = np.percentile(endwerte, 2.5)    # 95% Konfidenzintervall
conf95_upper_T = np.percentile(endwerte, 97.5)

print(f"\n--- {jahre} Jahre 3-ETF Portfolio Simulation (Empirische Regime-Klassifikation) ---")
print(f"📊 ETF-Allokation: {etf1_weight:.0%} / {etf2_weight:.0%} / {etf3_weight:.0%}")
rebalancing_desc = f"Alle {rebalance_frequency} Monate" if rebalance_frequency != 24 else "Alle 2 Jahre (tax-optimized)"
cash_flow_desc = " + Cash-Flow-Rebalancing" if cash_flow_rebalancing else ""
print(f"🔄 Rebalancing: {rebalancing_desc}{cash_flow_desc} (Schwelle: {rebalance_threshold:.0%})")
print(f"💰 ETF-Typ: {thesaurierend_etf_factor:.0%} thesaurierend (Dividenden steuerfrei reinvestiert)")
print(f"💰 Monatliche Sparrate: {monatliche_sparrate:.2f} €")
print(f"🎯 Regime-Parameter (MSM/Markov-Switching Studien):")
print(f"   • Bull: {regimes['bull']['mu']:.0%} Rendite, {regimes['bull']['vol_multiplier']:.1f}x Volatilität")
print(f"   • Normal: {regimes['normal']['mu']:.0%} Rendite, {regimes['normal']['vol_multiplier']:.1f}x Volatilität") 
print(f"   • Bear: {regimes['bear']['mu']:.0%} Rendite, {regimes['bear']['vol_multiplier']:.1f}x Volatilität")
print(f"📈 Durchschnittliche Inflation: {avg_inflation:.2%} pro Jahr")
print(f"🏦 Durchschnittlicher Zinssatz: {zinssatz_pfade.mean():.2%} pro Jahr")
print(f"💵 Gesamteinzahlungen (nominal): {gesamteinzahlungen_nominal:,.2f} €")
print(f"💵 Gesamteinzahlungen (inflationsbereinigt): {gesamteinzahlungen_real:,.2f} €")
print(f"💸 Durchschnittliche ETF-Kosten (TER): {ter_pfade[:, -1].mean():,.2f} €")
print(f"🏛️ Durchschnittliche Steuern: {steuer_pfade[:, -1].mean():,.2f} €")
print(f"💰 Gesamtkosten (TER + Steuern): {(ter_pfade[:, -1] + steuer_pfade[:, -1]).mean():,.2f} €")
print(f"💥 Black Swan Zusatzkosten: {(kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean():,.2f} €")
print(f"📊 ETF-Kostenquote (TER): {management_fee_rate:.1%} p.a.")
print(f"📈 Tax-Effizienz: Rebalancing alle {rebalance_frequency} Monate, Schwelle {rebalance_threshold:.0%}")
if cash_flow_rebalancing:
    print(f"💡 Cash-Flow-Rebalancing: Reduziert Turnover um ~40%")
print(f"🏦 Thesaurierende ETFs: {thesaurierend_etf_factor:.0%} der Dividenden steuerfrei")
print(f"💎 Durchschnittlicher Endwert (real): {endwerte.mean():,.2f} €")
print(f"💎 Durchschnittlicher Endwert (nominal): {endwerte_nominal.mean():,.2f} €")
print(f"📈 Durchschnittlicher Gewinn (real): {gewinn_verlust_real.mean():,.2f} €")
print(f"🎯 Median Endwert (real): {np.median(endwerte):,.2f} €")
print(f"📐 67%-Konfidenzintervall: {conf67_lower_T:,.2f} € bis {conf67_upper_T:,.2f} €")
print(f"⚠️ 75% VaR (reale Kaufkraft): {var75_T:,.2f} €")
print(f"🚨 95% VaR (reale Kaufkraft): {var95_T:,.2f} €")
print(f"💀 99% VaR (reale Kaufkraft): {var99_T:,.2f} €")

# Black Swan Event Analyse
print(f"\n🦢 --- BLACK SWAN EVENT ANALYSE ---")
hyperinflation_affected = np.any(hyperinflation_events, axis=1).sum()
crisis_affected = np.any(structural_crisis_events, axis=1).sum() 
tax_shock_affected = np.any(tax_shock_events, axis=1).sum()
personal_crisis_affected = np.any(personal_crisis_events, axis=1).sum()

print(f"🔥 Hyperinflation betroffen: {hyperinflation_affected:,} Simulationen ({hyperinflation_affected/num_sim:.1%})")
print(f"💥 Strukturkrise betroffen: {crisis_affected:,} Simulationen ({crisis_affected/num_sim:.1%})")
print(f"🏛️ Steuer-Schock betroffen: {tax_shock_affected:,} Simulationen ({tax_shock_affected/num_sim:.1%})")
print(f"🆘 Persönliche Krise betroffen: {personal_crisis_affected:,} Simulationen ({personal_crisis_affected/num_sim:.1%})")

# Vergleich: Endwerte mit vs ohne Black Swan Events
no_events_mask = (~np.any(hyperinflation_events, axis=1) & 
                  ~np.any(structural_crisis_events, axis=1) & 
                  ~np.any(tax_shock_events, axis=1) & 
                  ~np.any(personal_crisis_events, axis=1))

if np.any(no_events_mask):
    endwerte_no_events = endwerte[no_events_mask]
    print(f"\n📊 VERGLEICH: Mit vs Ohne Black Swan Events")
    print(f"   Ohne Events - Ø Endwert: {endwerte_no_events.mean():,.2f} €")
    print(f"   Mit Events - Ø Endwert: {endwerte.mean():,.2f} €")
    print(f"   Black Swan Impact: {((endwerte.mean() - endwerte_no_events.mean()) / endwerte_no_events.mean() * 100):+.1f}%")
    
    print(f"   Ohne Events - 95% VaR: {np.percentile(endwerte_no_events, 5):,.2f} €")
    print(f"   Mit Events - 95% VaR: {var95_T:,.2f} €")
    print(f"   VaR-Verschlechterung: {((var95_T - np.percentile(endwerte_no_events, 5)) / np.percentile(endwerte_no_events, 5) * 100):+.1f}%")

# --- Interaktive Visualisierung mit View-Auswahl ---
def show_main_view():
    """Zeigt die Hauptergebnisse der Monte-Carlo-Simulation"""
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: 1-Jahres-Renditen (Real vs Nominal)
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(renditen_1y_nominal, bins=50, alpha=0.6, label='Nominal', edgecolor='k')
    ax1.hist(renditen_1y, bins=50, alpha=0.6, label='Real (nach Inflation)', edgecolor='k')
    ax1.axvline(conf67_lower_1y, color='orange', linestyle=':', label=f"67% Konfidenz: {conf67_lower_1y:.2%}")
    ax1.axvline(conf67_upper_1y, color='orange', linestyle=':', label=f"bis {conf67_upper_1y:.2%}")
    ax1.axvline(var75_1y, color='purple', linestyle='-.', label=f"75% VaR = {var75_1y:.2%}")
    ax1.axvline(var95_1y, color='r', linestyle='--', label=f"95% VaR = {var95_1y:.2%}")
    ax1.axvline(var99_1y, color='g', linestyle='--', label=f"99% VaR = {var99_1y:.2%}")
    ax1.set_title("1-jährige Renditen: Real vs Nominal", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Rendite")
    ax1.set_ylabel("Häufigkeit")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Subplot 2: Portfolio-Entwicklung
    ax2 = plt.subplot(2, 2, 2)
    for i in range(50):
        ax2.plot(range(1, len(jahres_indizes)+1), kurse[i], alpha=0.6)
    ax2.set_title(f"Portfolio-Entwicklung über {jahre} Jahre", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Jahre")
    ax2.set_ylabel("Portfolio-Wert (€)")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} €'))

    # Subplot 3: Endwerte-Verteilung (Real vs Nominal)
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(endwerte_nominal, bins=50, alpha=0.6, label='Nominal', edgecolor='k')
    ax3.hist(endwerte, bins=50, alpha=0.6, label='Real (Kaufkraft)', edgecolor='k')
    ax3.axvline(conf67_lower_T, color='orange', linestyle=':', label=f"67% Konfidenz: {conf67_lower_T:.0f} €")
    ax3.axvline(conf67_upper_T, color='orange', linestyle=':', label=f"bis {conf67_upper_T:.0f} €")
    ax3.axvline(var75_T, color='purple', linestyle='-.', label=f"75% VaR = {var75_T:.0f} €")
    ax3.axvline(var95_T, color='r', linestyle='--', label=f"95% VaR = {var95_T:.0f} €")
    ax3.axvline(var99_T, color='g', linestyle='--', label=f"99% VaR = {var99_T:.0f} €")
    ax3.set_title(f"Endwerte nach {jahre} Jahren", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Portfolio-Wert (€)")
    ax3.set_ylabel("Häufigkeit")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} €'))
    ax3.legend(fontsize=10)

    # Subplot 4: Zusammenfassung der wichtigsten Kennzahlen
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # Vollständige Kennzahlen-Tabelle mit allen Konsolen-Ausgaben
    summary_data = [
        ["Kennzahl", "Wert"],
        ["ETF-Allokation", f"{etf1_weight:.0%}/{etf2_weight:.0%}/{etf3_weight:.0%}"],
        ["Monatliche Sparrate", f"{monatliche_sparrate:,.0f} €"],
        ["Anlagehorizont", f"{jahre} Jahre"],
        ["Gesamteinzahlungen (nominal)", f"{gesamteinzahlungen_nominal:,.0f} €"],
        ["Gesamteinzahlungen (real)", f"{gesamteinzahlungen_real:,.0f} €"],
        ["Ø Endwert (real)", f"{endwerte.mean():,.0f} €"],
        ["Ø Endwert (nominal)", f"{endwerte_nominal.mean():,.0f} €"],
        ["Ø Gewinn (real)", f"{gewinn_verlust_real.mean():,.0f} €"],
        ["Median Endwert (real)", f"{np.median(endwerte):,.0f} €"],
        ["67% Konfidenzintervall", f"{conf67_lower_T:,.0f} € - {conf67_upper_T:,.0f} €"],
        ["75% VaR", f"{var75_T:,.0f} €"],
        ["95% VaR", f"{var95_T:,.0f} €"],
        ["99% VaR", f"{var99_T:,.0f} €"],
        ["Ø Inflation p.a.", f"{avg_inflation:.2%}"],
        ["Ø Zinssatz p.a.", f"{zinssatz_pfade.mean():.2%}"],
        ["ETF-Kosten (TER)", f"{ter_pfade[:, -1].mean():,.0f} €"],
        ["Steuern (Dividenden+Gewinne)", f"{steuer_pfade[:, -1].mean():,.0f} €"],
        ["TER + Steuern", f"{(ter_pfade[:, -1] + steuer_pfade[:, -1]).mean():,.0f} €"],
        ["Black Swan Zusatzkosten", f"{(kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean():,.0f} €"],
        ["ETF-Kostenquote (TER) p.a.", f"{management_fee_rate:.1%}"]
    ]

    summary_table = ax4.table(cellText=summary_data, 
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.6, 0.4])

    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(12)
    summary_table.scale(1, 2)

    # Header formatieren
    for i in range(len(summary_data[0])):
        summary_table[(0, i)].set_facecolor('#2E7D32')
        summary_table[(0, i)].set_text_props(weight='bold', color='white')

    # Abwechselnde Zeilenfärbung
    for i in range(1, len(summary_data)):
        if i % 2 == 0:
            for j in range(len(summary_data[i])):
                summary_table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title("Wichtige Kennzahlen", fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Monte-Carlo-Simulation: Hauptergebnisse\n(Heston + Jump-Diffusion + Regime-Switching + Inflation)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def show_detail_view():
    """Zeigt die Detailanalysen der Monte-Carlo-Simulation"""
    fig = plt.figure(figsize=(18, 12))
    monate = np.arange(steps_monatlich + 1) / 12

    # Plot 1: Stochastische Zinssätze
    ax1 = plt.subplot(2, 2, 1)
    for i in range(50):
        ax1.plot(monate, zinssatz_pfade[i] * 100, alpha=0.6)
    ax1.axhline(theta_r * 100, color='red', linestyle='--', linewidth=2, label=f'Langfristig: {theta_r:.1%}')
    ax1.set_title("Stochastische Zinssätze (Vasicek-Modell)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Jahre")
    ax1.set_ylabel("Zinssatz (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Kumulierte Kosten und Steuern
    ax2 = plt.subplot(2, 2, 2)
    for i in range(20):  # Weniger Pfade für bessere Übersicht
        ax2.plot(monate, kosten_pfade[i], alpha=0.7, color='red', label='Gesamtkosten' if i == 0 else '')
        ax2.plot(monate, steuer_pfade[i], alpha=0.7, color='orange', label='Steuern' if i == 0 else '')
    ax2.set_title("Kumulierte Kosten & Steuern", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Jahre")
    ax2.set_ylabel("Kumulierte Kosten (€)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} €'))

    # Plot 3: Regime-Verteilung über Zeit
    ax3 = plt.subplot(2, 2, 3)
    regime_labels = ['Bull Market', 'Normal Market', 'Bear Market']
    regime_colors = ['green', 'blue', 'red']

    # Beispiel-Regime-Pfad (erste Simulation)
    beispiel_regimes = np.random.choice(3, size=steps_monatlich+1, p=[0.65, 0.30, 0.05])

    # Regime als Farbflächen
    for t in range(len(monate)-1):
        color = regime_colors[beispiel_regimes[t]]
        ax3.axvspan(monate[t], monate[t+1], alpha=0.7, color=color)

    ax3.set_title("Beispiel: Marktregime über Zeit", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Jahre")
    ax3.set_ylabel("Marktregime")
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(regime_labels)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Detaillierte Risikokennzahlen
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    risk_data = [
        ["Risikokennzahl", "1-Jahr (Real)", f"{jahre}-Jahre (Real)"],
        ["Mittelwert", f"{renditen_1y.mean():.2%}", f"{endwerte.mean():,.0f} €"],
        ["Standardabweichung", f"{renditen_1y.std():.2%}", f"{endwerte.std():,.0f} €"],
        ["Schiefe (Skewness)", f"{float(np.mean((renditen_1y - renditen_1y.mean())**3) / renditen_1y.std()**3):.3f}", 
         f"{float(np.mean((endwerte - endwerte.mean())**3) / endwerte.std()**3):.3f}"],
        ["Kurtosis (Excess)", f"{float(np.mean((renditen_1y - renditen_1y.mean())**4) / renditen_1y.std()**4 - 3):.3f}", 
         f"{float(np.mean((endwerte - endwerte.mean())**4) / endwerte.std()**4 - 3):.3f}"],
        ["75% VaR", f"{var75_1y:.2%}", f"{var75_T:,.0f} €"],
        ["95% VaR", f"{var95_1y:.2%}", f"{var95_T:,.0f} €"],
        ["99% VaR", f"{var99_1y:.2%}", f"{var99_T:,.0f} €"],
        ["95% Expected Shortfall", f"{es95_1y:.2%}", f"{endwerte[endwerte <= var95_T].mean():,.0f} €"],
        ["Verlustwahrscheinlichkeit", f"{(renditen_1y < 0).mean():.1%}", f"{(gewinn_verlust_real < 0).mean():.1%}"]
    ]

    risk_table = ax4.table(cellText=risk_data, 
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.4, 0.3, 0.3])

    risk_table.auto_set_font_size(False)
    risk_table.set_fontsize(11)
    risk_table.scale(1, 2)

    # Header formatieren
    for i in range(len(risk_data[0])):
        risk_table[(0, i)].set_facecolor('#1565C0')
        risk_table[(0, i)].set_text_props(weight='bold', color='white')

    # Abwechselnde Zeilenfärbung
    for i in range(1, len(risk_data)):
        if i % 2 == 0:
            for j in range(len(risk_data[i])):
                risk_table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title("Detaillierte Risikokennzahlen", fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Monte-Carlo-Simulation: Detailanalysen\n(Marktfaktoren + Kosten + Regime-Switching)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def interactive_view_navigator():
    """Interaktive Navigation zwischen Views mit Pfeiltasten"""
    import msvcrt  # Für Windows Tastatur-Input
    
    current_view = 0  # 0 = Hauptergebnisse, 1 = Detailanalysen
    views = ["Hauptergebnisse", "Detailanalysen"]
    
    def clear_screen():
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_navigation_header():
        print("\n" + "="*70)
        print("  MONTE-CARLO-SIMULATION - INTERAKTIVE NAVIGATION")
        print("="*70)
        print(f"\n📊 Aktuelle Ansicht: {views[current_view]}")
        print("\n🎯 Navigation:")
        print("  ←  Vorherige Ansicht")
        print("  →  Nächste Ansicht")
        print("  ESC  Beenden")
        print("="*70)
    
    def show_current_view():
        if current_view == 0:
            show_main_view()
        else:
            show_detail_view()
    
    # Zeige initial die erste View
    show_navigation_header()
    show_current_view()
    
    print(f"\n🔍 Drücken Sie ← → für Navigation oder ESC zum Beenden...")
    
    while True:
        try:
            # Warte auf Tastendruck
            key = msvcrt.getch()
            
            if key == b'\xe0':  # Extended key (Pfeiltasten)
                key = msvcrt.getch()  # Hole den eigentlichen Pfeil-Code
                
                if key == b'K':  # Linke Pfeiltaste
                    current_view = (current_view - 1) % len(views)
                    clear_screen()
                    show_navigation_header()
                    show_current_view()
                    print(f"\n🔍 Drücken Sie ← → für Navigation oder ESC zum Beenden...")
                    
                elif key == b'M':  # Rechte Pfeiltaste
                    current_view = (current_view + 1) % len(views)
                    clear_screen()
                    show_navigation_header()
                    show_current_view()
                    print(f"\n🔍 Drücken Sie ← → für Navigation oder ESC zum Beenden...")
                    
            elif key == b'\x1b':  # ESC-Taste
                print("\n\n✅ Navigation beendet.")
                break
                
        except KeyboardInterrupt:
            print("\n\n✅ Navigation beendet.")
            break
        except Exception as e:
            print(f"❌ Fehler: {e}")
class MatplotlibNavigationViews:
    """Provides interactive navigation between main and detailed views of Monte-Carlo simulation results using native Matplotlib tools."""
    
    def __init__(self, debug=False):
        self.current_view = 0
        self.views = ["Hauptergebnisse", "Detailanalysen", "Kosten & Black Swan"]
        self.fig = None
        self.tooltip = None
        self.debug = debug
        
    def create_main_plots(self):
        """Erstellt die Hauptergebnisse-Plots"""
        self.fig.clear()
        
        # Subplot 1: 1-Jahres-Renditen (Real vs Nominal)
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax1.hist(renditen_1y_nominal, bins=50, alpha=0.6, label='Nominal', edgecolor='k')
        ax1.hist(renditen_1y, bins=50, alpha=0.6, label='Real (nach Inflation)', edgecolor='k')
        ax1.axvline(conf67_lower_1y, color='orange', linestyle=':', label=f"67% Konfidenz: {conf67_lower_1y:.2%}")
        ax1.axvline(conf67_upper_1y, color='orange', linestyle=':', label=f"bis {conf67_upper_1y:.2%}")
        ax1.axvline(var95_1y, color='r', linestyle='--', label=f"95% VaR = {var95_1y:.2%}")
        ax1.set_title("1-jährige Renditen: Real vs Nominal", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Rendite")
        ax1.set_ylabel("Häufigkeit")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Subplot 2: Portfolio-Entwicklung
        ax2 = self.fig.add_subplot(2, 2, 2)
        for i in range(50):
            ax2.plot(range(1, len(jahres_indizes)+1), kurse[i], alpha=0.6)
        ax2.set_title(f"3-ETF Portfolio über {jahre} Jahre\n(50%/30%/20%, jährliches Rebalancing)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Jahre")
        ax2.set_ylabel("Portfolio-Wert (€)")
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} €'))

        # Subplot 3: Endwerte-Verteilung (Real vs Nominal)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax3.hist(endwerte_nominal, bins=50, alpha=0.6, label='Nominal', edgecolor='k')
        ax3.hist(endwerte, bins=50, alpha=0.6, label='Real (Kaufkraft)', edgecolor='k')
        ax3.axvline(conf67_lower_T, color='orange', linestyle=':', label=f"67% Konfidenz: {conf67_lower_T:.0f} €")
        ax3.axvline(conf67_upper_T, color='orange', linestyle=':', label=f"bis {conf67_upper_T:.0f} €")
        ax3.axvline(var95_T, color='r', linestyle='--', label=f"95% VaR = {var95_T:.0f} €")
        ax3.set_title(f"Endwerte nach {jahre} Jahren", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Portfolio-Wert (€)")
        ax3.set_ylabel("Häufigkeit")
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} €'))
        ax3.legend(fontsize=10)

        # Subplot 4: Zusammenfassung der wichtigsten Kennzahlen
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        # Kompakte Hauptkennzahlen-Tabelle (Kosten in separater View)
        summary_data = [
            ["Kennzahl", "Wert"],
            ["ETF-Allokation", f"{etf1_weight:.0%}/{etf2_weight:.0%}/{etf3_weight:.0%}"],
            ["Monatliche Sparrate", f"{monatliche_sparrate:,.0f} €"],
            ["Anlagehorizont", f"{jahre} Jahre"],
            ["Gesamteinzahlungen (real)", f"{gesamteinzahlungen_real:,.0f} €"],
            ["Ø Endwert (real)", f"{endwerte.mean():,.0f} €"],
            ["Ø Gewinn (real)", f"{gewinn_verlust_real.mean():,.0f} €"],
            ["Median Endwert (real)", f"{np.median(endwerte):,.0f} €"],
            ["67% Konfidenzbereich", f"{conf67_lower_T:,.0f} € - {conf67_upper_T:,.0f} €"],
            ["95% VaR (Worst Case)", f"{var95_T:,.0f} €"],
            ["99% VaR (Extremfall)", f"{var99_T:,.0f} €"],
            ["Ø Inflation p.a.", f"{avg_inflation:.2%}"],
            ["Ø Zinssatz p.a.", f"{zinssatz_pfade.mean():.2%}"],
            ["", "→ Kosten-Details: View 3"]
        ]

        summary_table = ax4.table(cellText=summary_data, 
                                 cellLoc='center',
                                 loc='center',
                                 colWidths=[0.6, 0.4])

        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(12)
        summary_table.scale(1, 1.8)

        # Header formatieren
        for i in range(len(summary_data[0])):
            summary_table[(0, i)].set_facecolor('#2E7D32')
            summary_table[(0, i)].set_text_props(weight='bold', color='white')

        # Abwechselnde Zeilenfärbung
        for i in range(1, len(summary_data)):
            if i % 2 == 0:
                for j in range(len(summary_data[i])):
                    summary_table[(i, j)].set_facecolor('#f0f0f0')

        # Tooltips für Haupttabelle
        self.add_table_tooltips(ax4, summary_table, self.get_main_tooltips())
        ax4.set_title("ETF-Portfolio Kennzahlen (Hover für Details)", fontsize=14, fontweight='bold', pad=20)
        
    def create_detail_plots(self):
        """Erstellt die Detailanalysen-Plots"""
        self.fig.clear()
        monate = np.arange(steps_monatlich + 1) / 12

        # Plot 1: Stochastische Zinssätze
        ax1 = self.fig.add_subplot(2, 2, 1)
        for i in range(50):
            ax1.plot(monate, zinssatz_pfade[i] * 100, alpha=0.6)
        ax1.axhline(theta_r * 100, color='red', linestyle='--', linewidth=2, label=f'Langfristig: {theta_r:.1%}')
        ax1.set_title("Stochastische Zinssätze (Vasicek-Modell)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Jahre")
        ax1.set_ylabel("Zinssatz (%)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Plot 2: Kumulierte Kosten und Steuern
        ax2 = self.fig.add_subplot(2, 2, 2)
        for i in range(20):  # Weniger Pfade für bessere Übersicht
            ax2.plot(monate, kosten_pfade[i], alpha=0.7, color='red', label='Gesamtkosten' if i == 0 else '')
            ax2.plot(monate, steuer_pfade[i], alpha=0.7, color='orange', label='Steuern' if i == 0 else '')
        ax2.set_title("Kumulierte Kosten & Steuern", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Jahre")
        ax2.set_ylabel("Kumulierte Kosten (€)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} €'))

        # Plot 3: Regime-Verteilung über Zeit
        ax3 = self.fig.add_subplot(2, 2, 3)
        regime_labels = ['Bull Market', 'Normal Market', 'Bear Market']
        regime_colors = ['green', 'blue', 'red']

        # Beispiel-Regime-Pfad (erste Simulation)
        beispiel_regimes = np.random.choice(3, size=steps_monatlich+1, p=[0.65, 0.30, 0.05])

        # Regime als Farbflächen
        for t in range(len(monate)-1):
            color = regime_colors[beispiel_regimes[t]]
            ax3.axvspan(monate[t], monate[t+1], alpha=0.7, color=color)

        ax3.set_title("Beispiel: Marktregime über Zeit", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Jahre")
        ax3.set_ylabel("Marktregime")
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(regime_labels)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Detaillierte Risikokennzahlen
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        # Berechne Black Swan Statistiken
        hyperinflation_affected = np.any(hyperinflation_events, axis=1).sum()
        crisis_affected = np.any(structural_crisis_events, axis=1).sum() 
        tax_shock_affected = np.any(tax_shock_events, axis=1).sum()
        personal_crisis_affected = np.any(personal_crisis_events, axis=1).sum()
        
        # Vergleich mit/ohne Black Swan Events
        no_events_mask = (~np.any(hyperinflation_events, axis=1) & 
                          ~np.any(structural_crisis_events, axis=1) & 
                          ~np.any(tax_shock_events, axis=1) & 
                          ~np.any(personal_crisis_events, axis=1))
        
        if np.any(no_events_mask):
            endwerte_no_events = endwerte[no_events_mask]
            black_swan_impact = ((endwerte.mean() - endwerte_no_events.mean()) / endwerte_no_events.mean() * 100)
        else:
            black_swan_impact = 0

        risk_data = [
            ["Risikokennzahl", "1-Jahr", f"{jahre}-Jahre (€)"],
            ["Mittelwert", f"{renditen_1y.mean():.2%}", f"{endwerte.mean():,.0f}"],
            ["Standardabweichung", f"{renditen_1y.std():.2%}", f"{endwerte.std():,.0f}"],
            ["67% Konfidenzbereich", f"{conf67_lower_1y:.1%} - {conf67_upper_1y:.1%}", f"{conf67_lower_T:,.0f} - {conf67_upper_T:,.0f}"],
            ["90% Konfidenzbereich", "-", f"{conf90_lower_T:,.0f} - {conf90_upper_T:,.0f}"],
            ["95% VaR (Worst Case)", f"{var95_1y:.2%}", f"{var95_T:,.0f}"],
            ["99% VaR (Extremfall)", f"{var99_1y:.2%}", f"{var99_T:,.0f}"],
            ["Expected Shortfall", f"{es95_1y:.2%}", f"{endwerte[endwerte <= var95_T].mean():,.0f}"],
            ["Verlustwahrscheinlichkeit", f"{(renditen_1y < 0).mean():.1%}", f"{(gewinn_verlust_real < 0).mean():.1%}"],
            ["", "", ""],
            ["→ Black Swan Details", "View 3", "→"]
        ]

        risk_table = ax4.table(cellText=risk_data, 
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.4, 0.3, 0.3])

        risk_table.auto_set_font_size(False)
        risk_table.set_fontsize(10)
        risk_table.scale(1, 1.4)

        # Header formatieren
        for i in range(len(risk_data[0])):
            risk_table[(0, i)].set_facecolor('#1565C0')
            risk_table[(0, i)].set_text_props(weight='bold', color='white')

        # Abwechselnde Zeilenfärbung
        for i in range(1, len(risk_data)):
            if i % 2 == 0:
                for j in range(len(risk_data[i])):
                    risk_table[(i, j)].set_facecolor('#f0f0f0')

        # Tooltips für Detailtabelle
        self.add_table_tooltips(ax4, risk_table, self.get_detail_tooltips())
        ax4.set_title("Risikokennzahlen (Hover für Details)", fontsize=13, fontweight='bold', pad=20)

    def create_cost_blackswan_plots(self):
        """Erstellt die Kosten- und Black Swan-Analyse-Plots"""
        self.fig.clear()
        monate = np.arange(steps_monatlich + 1) / 12

        # Plot 1: Kosten-Aufschlüsselung
        ax1 = self.fig.add_subplot(2, 2, 1)
        
        cost_categories = ['ETF-Kosten\n(TER)', 'Steuern', 'Black Swan\nZusatzkosten']
        cost_values = [
            ter_pfade[:, -1].mean(),
            steuer_pfade[:, -1].mean(), 
            (kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean()
        ]
        
        colors = ['#2E7D32', '#FF8F00', '#D32F2F']  # Grün, Orange, Rot
        bars = ax1.bar(cost_categories, cost_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Werte auf Balken anzeigen
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:,.0f}€', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax1.set_title("Kosten-Aufschlüsselung über 26 Jahre", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Kumulierte Kosten (€)")
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}€'))
        
        # Gesamtkosten als Text
        total_costs = sum(cost_values)
        ax1.text(0.5, 0.95, f'Gesamtkosten: {total_costs:,.0f}€', 
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=12, fontweight='bold')

        # Plot 2: Black Swan Event-Häufigkeiten
        ax2 = self.fig.add_subplot(2, 2, 2)
        
        # Berechne Black Swan Statistiken
        hyperinflation_affected = np.any(hyperinflation_events, axis=1).sum()
        crisis_affected = np.any(structural_crisis_events, axis=1).sum() 
        tax_shock_affected = np.any(tax_shock_events, axis=1).sum()
        personal_crisis_affected = np.any(personal_crisis_events, axis=1).sum()
        
        event_names = ['Hyper-\ninflation', 'Struktur-\nkrise', 'Steuer-\nSchock', 'Persönl.\nKrise']
        event_frequencies = [
            hyperinflation_affected/num_sim*100,
            crisis_affected/num_sim*100,
            tax_shock_affected/num_sim*100, 
            personal_crisis_affected/num_sim*100
        ]
        
        event_colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']
        bars2 = ax2.bar(event_names, event_frequencies, color=event_colors, alpha=0.8, edgecolor='black')
        
        # Prozente auf Balken
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_title("Black Swan Event-Häufigkeiten", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Betroffene Simulationen (%)")
        ax2.set_ylim(0, max(event_frequencies) * 1.2)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Kostenverlauf über Zeit
        ax3 = self.fig.add_subplot(2, 2, 3)
        
        # Mehrere repräsentative Kostenpfade zeigen
        for i in range(0, min(20, num_sim), 4):  # Jeden 4. der ersten 20
            ax3.plot(monate, ter_pfade[i], alpha=0.6, color='green', 
                    label='TER-Kosten' if i == 0 else '', linewidth=1.5)
            ax3.plot(monate, steuer_pfade[i], alpha=0.6, color='orange', 
                    label='Steuern' if i == 0 else '', linewidth=1.5)
            ax3.plot(monate, kosten_pfade[i], alpha=0.4, color='red', 
                    label='Gesamtkosten' if i == 0 else '', linewidth=1)
        
        ax3.set_title("Kostenverlauf über Zeit", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Jahre")
        ax3.set_ylabel("Kumulierte Kosten (€)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}€'))

        # Plot 4: Black Swan Impact-Tabelle
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        # Vergleich mit/ohne Black Swan Events
        no_events_mask = (~np.any(hyperinflation_events, axis=1) & 
                          ~np.any(structural_crisis_events, axis=1) & 
                          ~np.any(tax_shock_events, axis=1) & 
                          ~np.any(personal_crisis_events, axis=1))
        
        if np.any(no_events_mask):
            endwerte_no_events = endwerte[no_events_mask]
            black_swan_impact = ((endwerte.mean() - endwerte_no_events.mean()) / endwerte_no_events.mean() * 100)
            var_impact = ((var95_T - np.percentile(endwerte_no_events, 5)) / np.percentile(endwerte_no_events, 5) * 100)
        else:
            black_swan_impact = 0
            var_impact = 0

        impact_data = [
            ["Black Swan Impact", "Wert"],
            ["Betroffene Simulationen", f"{num_sim - np.sum(no_events_mask):,} von {num_sim:,}"],
            ["Hyperinflation-Ereignisse", f"{hyperinflation_affected:,} ({hyperinflation_affected/num_sim:.1%})"],
            ["Strukturkrisen", f"{crisis_affected:,} ({crisis_affected/num_sim:.1%})"],
            ["Steuer-Schocks", f"{tax_shock_affected:,} ({tax_shock_affected/num_sim:.1%})"],
            ["Persönliche Krisen", f"{personal_crisis_affected:,} ({personal_crisis_affected/num_sim:.1%})"],
            ["", ""],
            ["Ø Endwert ohne Events", f"{endwerte_no_events.mean() if np.any(no_events_mask) else 0:,.0f}€"],
            ["Ø Endwert mit Events", f"{endwerte.mean():,.0f}€"],
            ["Impact auf Ø Endwert", f"{black_swan_impact:+.1f}%"],
            ["95% VaR Verschlechterung", f"{var_impact:+.1f}%"],
            ["", ""],
            ["Zusätzliche Liquidationskosten", f"{(kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean():,.0f}€"]
        ]

        impact_table = ax4.table(cellText=impact_data, 
                               cellLoc='center',
                               loc='center',
                               colWidths=[0.6, 0.4])

        impact_table.auto_set_font_size(False)
        impact_table.set_fontsize(11)
        impact_table.scale(1, 1.8)

        # Header formatieren
        for i in range(len(impact_data[0])):
            impact_table[(0, i)].set_facecolor('#D32F2F')
            impact_table[(0, i)].set_text_props(weight='bold', color='white')

        # Abwechselnde Zeilenfärbung
        for i in range(1, len(impact_data)):
            if impact_data[i][0] == "":  # Leerzeilen
                for j in range(len(impact_data[i])):
                    impact_table[(i, j)].set_facecolor('#FFFFFF')
            elif i % 2 == 0:
                for j in range(len(impact_data[i])):
                    impact_table[(i, j)].set_facecolor('#FFEBEE')

        ax4.set_title("Black Swan Event-Impact Analyse", fontsize=14, fontweight='bold', pad=20)

    def get_main_tooltips(self):
        """Tooltip-Definitionen für die Haupttabelle (3-ETF Portfolio)"""
        return {
            1: "Gewichtung der 3 ETFs: World Aktien/Emerging Markets/Anleihen mit jährlichem Rebalancing",
            2: "Monatlich investierter Betrag in das 3-ETF Portfolio",
            3: "Gesamter Anlagezeitraum für die ETF-Simulation mit Heston-Modell und Black Swan Events", 
            4: "Inflationsbereinigte Summe aller ETF-Einzahlungen über die Laufzeit (reale Kaufkraft)",
            5: "Durchschnittlicher realer Endwert nach Inflation, allen Kosten und Black Swan Events",
            6: "Durchschnittlicher realer Gewinn nach Abzug aller Kosten, Steuern und Inflation",
            7: "Median (50%-Quantil) des realen Endwerts - robuster gegen Extremwerte als der Mittelwert",
            8: "Konfidenzbereich: Mit 67% Wahrscheinlichkeit liegt der Endwert in diesem Bereich (±1σ)",
            9: "Value at Risk: In 5% der Fälle liegt der Endwert unter diesem Wert (Risikoszenario)",
            10: "Value at Risk: In 1% der Fälle liegt der Endwert unter diesem Wert (Extremszenario)",
            11: "Durchschnittliche jährliche Inflationsrate über die gesamte Simulationsperiode",
            12: "Durchschnittlicher risikofreier Zinssatz (Vasicek-Modell mit stochastischen Zinsen)",
            13: "Detaillierte Kostenaufschlüsselung und Black Swan Analyse in der dritten View verfügbar"
        }
    
    def get_detail_tooltips(self):
        """Tooltip-Definitionen für die Detailtabelle"""
        return {
            1: "Durchschnittliche erwartete reale Rendite pro Jahr (nach Inflation)",
            2: "Maß für die Schwankungsbreite der Renditen (Volatilität/Risiko)",
            3: "Konfidenzintervall: Bereich, in dem die Werte mit 67%iger Wahrscheinlichkeit liegen (±1σ)",
            4: "Konfidenzintervall: Bereich, in dem die Werte mit 90%iger Wahrscheinlichkeit liegen", 
            5: "Value at Risk: In 5% der Fälle ist das Ergebnis schlechter als dieser Wert (Risikoszenario)",
            6: "Value at Risk: In 1% der Fälle ist das Ergebnis schlechter als dieser Wert (Extremszenario)",
            7: "Expected Shortfall: Durchschnittlicher Verlust in den schlechtesten 5% der Fälle",
            8: "Wahrscheinlichkeit eines Verlusts (realer Endwert kleiner als reale Einzahlungen)",
            11: "Anzahl Simulationen mit Hyperinflations-Perioden (12% Inflation über 3 Jahre)",
            12: "Anzahl Simulationen mit Strukturkrisen (50% Markt-Drawdown über 2 Jahre)", 
            13: "Anzahl Simulationen mit Steuer-Schocks (45% Kapitalertragssteuer + 1% Vermögenssteuer)",
            14: "Anzahl Simulationen mit persönlichen Liquidationskrisen (30% Notverkauf)",
            15: "Durchschnittlicher negativer Einfluss aller Black Swan Events auf den Endwert"
        }
    
    def add_table_tooltips(self, ax, table, tooltips_dict):
        """Fügt interaktive Tooltips zu einer Tabelle hinzu"""
        def on_hover(event):
            if event.inaxes != ax:
                return
            
            # Finde die Zelle unter dem Mauszeiger
            for (row, col), cell in table.get_celld().items():
                if row == 0:  # Skip header row
                    continue
                    
                bbox = cell.get_window_extent(renderer=self.fig.canvas.get_renderer())
                
                # Transformiere Koordinaten
                bbox_data = ax.transData.inverted().transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
                
                if (bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1):
                    # Tooltip anzeigen
                    if row in tooltips_dict:
                        self.show_tooltip(event, tooltips_dict[row])
                        return
            
            # Tooltip verstecken wenn nicht über einer Zelle
            self.hide_tooltip()
        
        # Event-Handler verbinden
        self.fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def show_tooltip(self, event, text):
        """Zeigt einen Tooltip an der Mausposition"""
        if self.tooltip:
            self.tooltip.remove()
        
        # Tooltip-Box erstellen
        self.tooltip = self.fig.text(
            event.x / self.fig.dpi / self.fig.get_size_inches()[0], 
            event.y / self.fig.dpi / self.fig.get_size_inches()[1],
            text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9, edgecolor="gray"),
            fontsize=10,
            wrap=True,
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=self.fig.transFigure,
            zorder=1000
        )
        self.fig.canvas.draw_idle()
    
    def hide_tooltip(self):
        """Versteckt den aktuellen Tooltip"""
        if self.tooltip:
            self.tooltip.remove()
            self.tooltip = None
            self.fig.canvas.draw_idle()
        
    def update_view(self):
        """Aktualisiert die aktuelle View"""
        if self.current_view == 0:
            self.create_main_plots()
            title = 'Monte-Carlo-Simulation: Hauptergebnisse'
        elif self.current_view == 1:
            self.create_detail_plots()
            title = 'Monte-Carlo-Simulation: Detailanalysen'
        else:
            self.create_cost_blackswan_plots()
            title = 'Monte-Carlo-Simulation: Kosten & Black Swan Analyse'
        
        self.fig.suptitle(f'{title}\n(Heston + Jump-Diffusion + Regime-Switching + Inflation)', 
                         fontsize=16, fontweight='bold')
        self.fig.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Navigation-Hinweis hinzufügen
        nav_text = f"View: {self.views[self.current_view]} | ← → Pfeiltasten | ESC = Beenden"
        self.fig.text(0.5, 0.02, nav_text, ha='center', va='bottom', fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        self.fig.canvas.draw()
        
    def on_key_press(self, event):
        """Event-Handler für Tastatureingaben (native Matplotlib)"""
        if self.debug:
            print(f"Key pressed: '{event.key}'")  # Debug output
        
        if event.key in ['right', 'down', 'n', 'space']:
            # Nächste View (multiple key options)
            if self.debug:
                print(f"Switching to next view from {self.current_view}")
            self.current_view = (self.current_view + 1) % len(self.views)
            self.update_view()
        elif event.key in ['left', 'up', 'p', 'backspace']:
            # Vorherige View (multiple key options)
            if self.debug:
                print(f"Switching to previous view from {self.current_view}")
            self.current_view = (self.current_view - 1) % len(self.views)
            self.update_view()
        elif event.key == 'escape':
            # Beenden
            if self.debug:
                print("Closing figure...")
            plt.close(self.fig)
        else:
            if self.debug:
                print(f"Unhandled key: '{event.key}'")
            self.current_view = (self.current_view - 1) % len(self.views)
            self.update_view()
            
    def on_mouse_click(self, event):
        """Event-Handler für Mausklicks - Alternative Navigation"""
        if event.button == 1:  # Left mouse button
            # Check if click is in navigation area (bottom 10% of figure)
            if event.y < 0.1 * self.fig.get_size_inches()[1] * self.fig.dpi:
                if event.x < 0.5 * self.fig.get_size_inches()[0] * self.fig.dpi:
                    # Left half - previous view
                    print("Mouse click: Previous view")
                    self.current_view = (self.current_view - 1) % len(self.views)
                    self.update_view()
                else:
                    # Right half - next view
                    print("Mouse click: Next view")
                    self.current_view = (self.current_view + 1) % len(self.views)
                    self.update_view()

    def start_interactive_navigation(self):
        """Startet die interaktive Navigation mit nativen Matplotlib-Tools"""
        print("\n🎯 Starte interaktive Navigation mit Matplotlib...")
        print("📋 Steuerung:")
        print("   ← → ↑ ↓ Pfeiltasten: Views wechseln (3 Views verfügbar)")
        print("   N / Space: Nächste View")
        print("   P / Backspace: Vorherige View")
        print("   Mausklick unten links/rechts: Navigation")
        print("   ESC: Beenden")
        print("   Plus: Alle nativen Matplotlib-Werkzeuge!")
        print("\n💡 Tipp: Klicken Sie erst einmal in die Grafik, um den Fokus zu setzen.")
        
        # Figur erstellen
        self.fig = plt.figure(figsize=(18, 12))
        
        # Event-Handler verbinden (native Matplotlib)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        # Erste View anzeigen
        self.update_view()
        
        # Interactive mode für bessere Responsiveness
        plt.ion()
        plt.show()
        
        # Keep the window alive
        try:
            while plt.get_fignums():  # While figure window is open
                plt.pause(0.1)
        except KeyboardInterrupt:
            print("\n✅ Navigation beendet.")
        finally:
            plt.ioff()


def show_cost_blackswan_view():
    """Zeigt die Kosten- und Black Swan-Analyse der Monte-Carlo-Simulation"""
    fig = plt.figure(figsize=(18, 12))
    monate = np.arange(steps_monatlich + 1) / 12

    # Plot 1: Kosten-Aufschlüsselung
    ax1 = plt.subplot(2, 2, 1)
    
    cost_categories = ['ETF-Kosten\n(TER)', 'Steuern', 'Black Swan\nZusatzkosten']
    cost_values = [
        ter_pfade[:, -1].mean(),
        steuer_pfade[:, -1].mean(), 
        (kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean()
    ]
    
    colors = ['#2E7D32', '#FF8F00', '#D32F2F']  # Grün, Orange, Rot
    bars = ax1.bar(cost_categories, cost_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Werte auf Balken anzeigen
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:,.0f}€', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_title("Kosten-Aufschlüsselung über 26 Jahre", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Kumulierte Kosten (€)")
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}€'))
    
    # Gesamtkosten als Text
    total_costs = sum(cost_values)
    ax1.text(0.5, 0.95, f'Gesamtkosten: {total_costs:,.0f}€', 
            transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=12, fontweight='bold')

    # Plot 2: Black Swan Event-Häufigkeiten
    ax2 = plt.subplot(2, 2, 2)
    
    hyperinflation_affected = np.any(hyperinflation_events, axis=1).sum()
    crisis_affected = np.any(structural_crisis_events, axis=1).sum() 
    tax_shock_affected = np.any(tax_shock_events, axis=1).sum()
    personal_crisis_affected = np.any(personal_crisis_events, axis=1).sum()
    
    event_names = ['Hyper-\ninflation', 'Struktur-\nkrise', 'Steuer-\nSchock', 'Persönl.\nKrise']
    event_frequencies = [
        hyperinflation_affected/num_sim*100,
        crisis_affected/num_sim*100,
        tax_shock_affected/num_sim*100, 
        personal_crisis_affected/num_sim*100
    ]
    
    event_colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']
    bars2 = ax2.bar(event_names, event_frequencies, color=event_colors, alpha=0.8, edgecolor='black')
    
    # Prozente auf Balken
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_title("Black Swan Event-Häufigkeiten", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Betroffene Simulationen (%)")
    ax2.set_ylim(0, max(event_frequencies) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Kostenverlauf über Zeit
    ax3 = plt.subplot(2, 2, 3)
    
    # Durchschnittlicher Kostenverlauf
    ter_mean = np.mean(ter_pfade, axis=0)
    steuer_mean = np.mean(steuer_pfade, axis=0)
    kosten_mean = np.mean(kosten_pfade, axis=0)
    
    ax3.plot(monate, ter_mean, color='green', linewidth=3, label='TER-Kosten (Ø)', alpha=0.8)
    ax3.plot(monate, steuer_mean, color='orange', linewidth=3, label='Steuern (Ø)', alpha=0.8)
    ax3.plot(monate, kosten_mean, color='red', linewidth=2, label='Gesamtkosten (Ø)', alpha=0.6)
    
    # Konfidenzbänder
    ter_upper = np.percentile(ter_pfade, 75, axis=0)
    ter_lower = np.percentile(ter_pfade, 25, axis=0)
    ax3.fill_between(monate, ter_lower, ter_upper, alpha=0.2, color='green')
    
    ax3.set_title("Kostenverlauf über Zeit", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Jahre")
    ax3.set_ylabel("Kumulierte Kosten (€)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}€'))

    # Plot 4: Black Swan Impact-Zusammenfassung
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Impact-Berechnung
    no_events_mask = (~np.any(hyperinflation_events, axis=1) & 
                      ~np.any(structural_crisis_events, axis=1) & 
                      ~np.any(tax_shock_events, axis=1) & 
                      ~np.any(personal_crisis_events, axis=1))
    
    if np.any(no_events_mask):
        endwerte_no_events = endwerte[no_events_mask]
        black_swan_impact = ((endwerte.mean() - endwerte_no_events.mean()) / endwerte_no_events.mean() * 100)
    else:
        black_swan_impact = 0

    impact_summary = f"""
    🎆 BLACK SWAN EVENT-ZUSAMMENFASSUNG
    
    🔥 Hyperinflation: {hyperinflation_affected:,} Simulationen ({hyperinflation_affected/num_sim:.1%})
    💥 Strukturkrisen: {crisis_affected:,} Simulationen ({crisis_affected/num_sim:.1%})
    🏛️ Steuer-Schocks: {tax_shock_affected:,} Simulationen ({tax_shock_affected/num_sim:.1%})
    🆘 Pers. Krisen: {personal_crisis_affected:,} Simulationen ({personal_crisis_affected/num_sim:.1%})
    
    📊 PERFORMANCE-IMPACT:
    • Ohne Events: {endwerte_no_events.mean() if np.any(no_events_mask) else 0:,.0f}€
    • Mit Events: {endwerte.mean():,.0f}€
    • Durchschn. Impact: {black_swan_impact:+.1f}%
    
    💰 ZUSATZKOSTEN:
    • Liquidationsstrafen: {(kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean():,.0f}€
    • Anteil an Gesamtkosten: {(kosten_pfade[:, -1] - ter_pfade[:, -1] - steuer_pfade[:, -1]).mean() / kosten_pfade[:, -1].mean() * 100:.1f}%
    """
    
    ax4.text(0.05, 0.95, impact_summary, transform=ax4.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=12, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", alpha=0.9, edgecolor="#D32F2F"))

    plt.suptitle('Monte-Carlo-Simulation: Kosten & Black Swan Analyse\n(Detaillierte Aufschlüsselung aller Kostenfaktoren)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def show_simple_menu():
    """Menü zur Auswahl der Darstellungsart"""
    print("\n" + "="*60)
    print("  MONTE-CARLO-SIMULATION - VIEW AUSWAHL")
    print("="*60)
    print("\n📊 Verfügbare Ansichten:")
    print("  [1] Hauptergebnisse (statisch)")
    print("  [2] Detailanalysen (statisch)")
    print("  [3] Kosten & Black Swan (statisch)")
    print("  [4] Interaktive Navigation (alle 3 Views)")
    print("  [0] Beenden")
    
    while True:
        try:
            choice = input("\n🔍 Ihre Auswahl (0-4): ").strip()
            
            if choice == '1':
                print("\n📈 Zeige Hauptergebnisse...")
                show_main_view()
                break
            elif choice == '2':
                print("\n🔬 Zeige Detailanalysen...")
                show_detail_view()
                break
            elif choice == '3':
                print("\n💰 Zeige Kosten & Black Swan Analyse...")
                show_cost_blackswan_view()
                break
            elif choice == '4':
                navigator = MatplotlibNavigationViews()
                navigator.start_interactive_navigation()
                break
            elif choice == '0':
                print("\n✅ Programm beendet.")
                break
            else:
                print("❌ Ungültige Eingabe. Bitte wählen Sie 0-4.")
        except KeyboardInterrupt:
            print("\n\n✅ Programm beendet.")
            break
        except Exception as e:
            print(f"❌ Fehler: {e}")

# Starte das Menü
show_simple_menu()
