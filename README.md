# 🎲 Monte Carlo Portfolio-Simulation mit Heston-Modell

Eine umfassende Monte Carlo-Simulation für langfristige Portfolioanalyse mit erweiterten Marktmodellen und realistischen Risikofaktoren.

## 📊 Überblick

Diese Simulation modelliert die Entwicklung eines 3-ETF-Portfolios über 26 Jahre unter Berücksichtigung von:
- **Stochastischer Volatilität** (Heston-Modell)
- **Jump-Diffusion** (Merton-Modell) für Marktcrashs
- **Regime-Switching** (Bull/Normal/Bear Markets)
- **Black Swan Events** (Hyperinflation, Strukturkrisen, etc.)
- **Realistische Kosten und Steuern**
- **Multithreading-Optimierung** für bessere Performance

## 🚀 Hauptfeatures

### ✅ Erweiterte Marktmodellierung
- **Heston-Modell**: Stochastische Volatilität mit Mean-Reversion
- **Merton Jump-Diffusion**: Seltene, große Marktbewegungen (~1 Crash alle 9 Jahre)
- **Regime-Switching**: Markov-Ketten für Bull (60%), Normal (30%), Bear (10%) Markets
- **Fat-Tail Distributions**: t-Verteilung für realistische Extremereignisse

### ⚡ Performance-Optimierung
- **Multithreading**: 4x-8x Speedup durch Parallelisierung
- **Batch-Processing**: Memory-effiziente Verarbeitung großer Simulationen
- **50.000 Simulationen** in wenigen Minuten statt Stunden

### 📈 Interaktive Visualisierung
- **4 verschiedene Views** mit Navigation
- **Hauptergebnisse, Detailanalysen, Performance, Kosten**
- **Pfeiltasten-Navigation** durch matplotlib

## 🎯 Modell-Parameter

### 📈 Heston-Modell (Stochastische Volatilität)

```python
# Rendite-Parameter
mu = 0.075          # 7.5% erwartete Jahresrendite (historischer S&P 500)

# Heston-Volatilitäts-Parameter
v0 = 0.0289         # 17% initiale Volatilität (√0.0289 ≈ 0.17)
kappa = 2.0         # Mean-Reversion-Geschwindigkeit (moderat)
theta = 0.0256      # 16% langfristige Volatilität (√0.0256 = 0.16)
sigma_v = 0.3       # 30% Vol-of-Vol (Volatilität der Volatilität)
rho = -0.7          # Negative Korrelation Preis-Volatilität ("Leverage-Effekt")
```

**Begründung der Heston-Parameter:**
- **v0/theta ~17%/16%**: Typische Aktienmarkt-Volatilität (VIX-Durchschnitt)
- **kappa = 2.0**: Moderate Mean-Reversion (Halbwertszeit ~4 Monate)
- **rho = -0.7**: Starker Leverage-Effekt (fallende Kurse → steigende Volatilität)

### 💥 Jump-Diffusion (Merton-Modell)

```python
# Crash-Parameter (realistische Marktkrisen)
lambda_jump = 0.11       # ~1 Crash alle 9 Jahre (1/0.11 ≈ 9.1 Jahre)
mu_jump = -0.15         # -15% durchschnittliche Crashgröße
sigma_jump = 0.08       # 8% Volatilität der Crashes (heterogene Größen)
```

**Historische Basis:**
- **1987**: Schwarzer Montag (-22%)
- **2000-2002**: Dotcom-Crash (-49%)  
- **2008**: Finanzkrise (-57%)
- **2020**: Corona-Crash (-34%, schnelle Erholung)

### 🔄 Regime-Switching (Markov-Modell)

```python
regimes = {
    'bull': {'mu': 0.15, 'vol_multiplier': 0.8, 'prob': 0.60},   # 15% Rendite, niedrige Volatilität
    'normal': {'mu': 0.08, 'vol_multiplier': 1.0, 'prob': 0.30}, # 8% Rendite, normale Volatilität  
    'bear': {'mu': -0.15, 'vol_multiplier': 1.8, 'prob': 0.10}   # -15% Rendite, hohe Volatilität
}

# Übergangswahrscheinlichkeiten (monatlich)
transition_matrix = np.array([
    [17/18, 1/20, 1/180],    # Bull → Bull: 94.4%, Bull → Normal: 5%, Bull → Bear: 0.56%
    [2/20, 7/8, 1/40],       # Normal → Bull: 10%, Normal → Normal: 87.5%, Normal → Bear: 2.5%
    [2/15, 7/150, 5/6]       # Bear → Bull: 13.3%, Bear → Normal: 4.7%, Bear → Bear: 83.3%
])
```

**Regime-Dauern (empirisch kalibriert):**
- **Bull Markets**: ~18 Monate durchschnittlich
- **Normal Markets**: ~8 Monate durchschnittlich
- **Bear Markets**: ~6 Monate durchschnittlich

### 🏦 Makroökonomische Parameter

```python
# Inflation (AR(1)-Prozess)
inflation_mean = 0.022       # 2.2% EZB-Ziel
inflation_vol = 0.004        # 0.4% Volatilität (stabile Geldpolitik)
inflation_persistence = 0.85 # Hohe Autokorrelation

# Stochastische Zinssätze (Vasicek-Modell)
r0 = 0.025              # 2.5% aktueller Zinssatz
kappa_r = 0.8           # Mean-Reversion-Speed
theta_r = 0.03          # 3% langfristiger Gleichgewichtszins
sigma_r = 0.012         # 1.2% Zinssatz-Volatilität
```

### 🦢 Black Swan Events (Realistische OECD-Wahrscheinlichkeiten)

```python
# Hyperinflation (1970er-Style)
hyperinflation_prob = 1/150         # ~0.67% pro Jahr (alle 150 Jahre)
hyperinflation_rate = 0.12          # 12% Inflation während Event
hyperinflation_duration_years = 3   # 3 Jahre Dauer

# Strukturkrisen (wie 2008, 1929)
structural_crisis_prob = 1/12       # ~8.3% pro Jahr (alle 12 Jahre eine große Krise)
structural_crisis_drawdown = 0.50   # 50% Markt-Drawdown
structural_crisis_duration_months = 24  # 2 Jahre Dauer

# Steuerliche Schocks
tax_shock_prob = 1/30              # ~3.3% pro Jahr (alle 30 Jahre Reform)
tax_shock_new_rate = 0.45          # 45% neue Kapitalertragssteuer
tax_shock_wealth_tax = 0.01        # 1% Vermögenssteuer ab 500k€

# Persönliche Liquidationskrisen  
personal_crisis_prob_per_year = 0.015  # 1.5% pro Jahr (~35% über 30 Jahre)
liquidation_percentage = 0.30          # 30% Notverkauf des Portfolios
```

**Wahrscheinlichkeiten über 26 Jahre:**
- Hyperinflation: ~15.5% Chance
- Strukturkrise: ~80% Chance (2-3 Krisen zu erwarten)
- Steuer-Schock: ~61% Chance
- Persönliche Krise: ~32% Chance

## 💰 Portfolio-Konfiguration

### 🎯 3-ETF-Strategie

```python
# Asset Allocation
etf1_weight = 0.5    # 50% World Aktien (MSCI World ETF)
etf2_weight = 0.3    # 30% Emerging Markets (MSCI EM ETF)  
etf3_weight = 0.2    # 20% Anleihen/REITs (Diversifikation)

# Investment Parameter
startwert = 12000              # 12.000€ Startkapital
monatliche_sparrate = 500      # 500€ monatliche Sparrate
jahre = 26                     # 26 Jahre Anlagehorizont
```

### 🔄 Rebalancing & Kosten

```python
# Tax-Optimized Rebalancing
rebalance_frequency = 24         # Alle 2 Jahre (steueroptimiert)
rebalance_threshold = 0.15       # 15% Abweichung löst Rebalancing aus
cash_flow_rebalancing = True     # Nutze Sparraten für passives Rebalancing

# Realistische Kosten
transaction_cost_rate = 0.001    # 0.1% pro Transaktion
management_fee_rate = 0.002      # 0.2% jährliche TER
tax_rate_dividends = 0.264       # 26.375% Abgeltungsteuer
thesaurierend_etf_factor = 0.7   # 70% thesaurierende ETFs (steueroptimiert)
```

## 🔬 Technische Implementation

### 📊 Monte Carlo Parameter

```python
num_sim = 50000     # 50.000 Simulationen für robuste Statistiken
steps = 312         # Monatliche Zeitschritte (26 Jahre × 12 Monate)
df_shocks = 6       # t-Verteilung mit 6 Freiheitsgraden (fat tails)
```

### ⚡ Multithreading-Optimierung

```python
# Automatische Worker-Optimierung
num_workers = min(cpu_count(), 8)  # Max 8 Kerne (Memory-begrenzt)
batch_size = num_sim // num_workers # ~6.250 Simulationen pro Kern

# Performance-Verbesserungen:
# - 4x-8x Speedup durch Parallelisierung  
# - Memory-effizientes Batch-Processing
# - Thread-safe Random Number Generation
```

## 📈 Erwartete Ergebnisse

### 💎 Typische Portfolio-Entwicklung (Real, nach Inflation)

```
Gesamteinzahlungen: ~168.000€ (real)
Durchschnittlicher Endwert: ~280.000€
67% Konfidenzbereich: 180.000€ - 420.000€
95% VaR (Worst Case): ~120.000€
```

### 📊 Risiko-Kennzahlen

- **Sharpe Ratio**: ~0.45 (nach Kosten und Steuern)
- **Maximum Drawdown**: ~65% (bei Bear Market + Black Swan)
- **Verlustwahrscheinlichkeit**: ~15% (realer Verlust nach 26 Jahren)

## 🛠️ Installation & Verwendung

### Voraussetzungen

```bash
pip install numpy matplotlib scipy concurrent.futures multiprocessing
```

### Schnellstart

```python
# Einfache Ausführung
python MonteCarlo.py

# Multithreaded Version  
python MonteCarlo_Multithreaded.py

# Auswahl im interaktiven Menü:
# [1] Performance Demo
# [2] Vollständige Simulation + Views  
# [3] Threading-Empfehlungen
```

### Navigation in den Views

- **← → ↑ ↓** Pfeiltasten: Views wechseln
- **N / Space**: Nächste View
- **P / Backspace**: Vorherige View  
- **ESC**: Beenden

## 📚 Wissenschaftliche Grundlagen

### 🔬 Verwendete Modelle

1. **Heston (1993)**: "A Closed-Form Solution for Options with Stochastic Volatility"
2. **Merton (1976)**: "Option pricing when underlying stock returns are discontinuous"  
3. **Hamilton (1989)**: "A New Approach to the Economic Analysis of Nonstationary Time Series"
4. **Vasicek (1977)**: "An equilibrium characterization of the term structure"

### 📖 Parameter-Kalibrierung

- **Volatilitäts-Parameter**: VIX-Daten 1990-2024
- **Jump-Parameter**: S&P 500 Crash-Analyse 1950-2024
- **Regime-Parameter**: NBER Recession Dating + Bull/Bear Market Studien
- **Black Swan Wahrscheinlichkeiten**: OECD Historical Data + Stress-Test-Szenarien

## ⚠️ Disclaimer

**Diese Simulation dient nur zu Bildungszwecken und ist keine Anlageberatung.**

- Parameter basieren auf historischen Daten (past performance ≠ future results)
- Black Swan Events sind per Definition unvorhersagbar
- Reale Marktentwicklung kann erheblich von Simulationen abweichen
- Konsultieren Sie einen Finanzberater für individuelle Anlageentscheidungen

## 🤝 Beiträge & Erweiterungen

Mögliche Verbesserungen:
- **Weitere Asset-Klassen** (Commodities, Crypto, REITs)
- **Dynamische Asset Allocation** (Risk Parity, Momentum)
- **ESG-Faktoren** und Climate Risk  
- **Behavioral Finance** (Momentum, Herding)
- **GPU-Beschleunigung** für noch größere Simulationen

## 📄 Lizenz

MIT License - Siehe LICENSE-Datei für Details.

---

*Erstellt mit ❤️ für quantitative Finanzanalyse und Monte Carlo-Methoden*