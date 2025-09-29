import numpy as np

# Test the new transition matrix
transition_matrix = np.array([
    [0.92, 0.070, 0.010],
    [0.30, 0.60, 0.10],
    [0.20, 0.65, 0.15]
])

eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
steady_state_index = np.argmax(np.real(eigenvalues))
steady_state = np.real(eigenvectors[:, steady_state_index])
steady_state = steady_state / steady_state.sum()

print(f'📊 Neue Transition Matrix - Steady State:')
print(f'   Bull Market: {steady_state[0]:.1%} (Ziel: 55-65%)')
print(f'   Normal Market: {steady_state[1]:.1%} (Ziel: 25-35%)')  
print(f'   Bear Market: {steady_state[2]:.1%} (Ziel: 5-15%)')

print(f'\n🎯 Regime-Dauern:')
print(f'   Bull: {1/(1-0.92):.1f} Monate (Ziel: 12-24)')
print(f'   Normal: {1/(1-0.60):.1f} Monate (Ziel: 6-12)')
print(f'   Bear: {1/(1-0.15):.1f} Monate (Ziel: 6-12)')

print(f'\n💥 Jump-Parameter:')
print(f'   Frequenz: 1/{0.11:.2f} = {1/0.11:.1f} Jahre zwischen Crashes')
print(f'   Crash-Größe: {-0.15:.0%} ± {0.08:.0%}')