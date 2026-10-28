"""Teste de sanidade do avaliador: usa os próprios segmentos Hough de (b) como gabarito.

Os segmentos foram obtidos da máscara branca B, então o recall de B deve ser alto.

Uso: python3 scripts/diagnostico/diag_sanidade_avaliador.py
"""
import sys, os, json, tempfile, subprocess

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# Segmentos Hough de diag_b2_perp.py (fornecidos como fixos)
# formato: [[x1,y1],[x2,y2]]
SEGS = {
    'bitbots': [
        [[77, 333], [314, 228]],
        [[63, 351], [285, 464]],
        [[281, 428], [462, 265]],
    ],
    'eliza': [
        [[71, 246], [303, 317]],
        [[413, 359], [635, 426]],
        [[428, 355], [604, 406]],
    ],
}

# FWHM mediano por segmento (de diag_b2_perp.py):
#   bitbots: 16.2, 24.2, 26.4 px → median=24.2 → d_base=12
#   eliza:    6.5, 10.8,  9.7 px → median= 9.7 → d_base= 5
# Nota: o usuário indicou d≈10 para bitbots; aqui usamos d=12 (medido) e d=10 (estimado)
# para transparência, testamos ambos: d_base e d_base+4.

D_PARAMS = {
    'bitbots': {'fwhm_med': 24.2, 'd_base': 12, 'd_tol': 16},
    'eliza':   {'fwhm_med':  9.7, 'd_base':  5, 'd_tol':  9},
}

script = os.path.join(ROOT, 'scripts', 'avaliar_mascara_branco.py')

for nome in ['bitbots', 'eliza']:
    polys = SEGS[nome]
    dp    = D_PARAMS[nome]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False, encoding='utf-8') as f:
        tmp = f.name
        json.dump({nome: polys}, f, indent=2)

    print("=" * 70)
    print(f"{nome}: {len(polys)} segmentos Hough como gabarito")
    print(f"  FWHM mediano dos segmentos: {dp['fwhm_med']:.1f} px")
    print(f"  d_base = round(FWHM_med/2) = {dp['d_base']}  │  d_tol = d_base+4 = {dp['d_tol']}")
    print("=" * 70)

    for d in [dp['d_base'], dp['d_tol']]:
        print(f"\n── d={d} px (diam={2*d+1}px) ──")
        result = subprocess.run(
            [sys.executable, script, nome, f'--d={d}',
             f'--gabarito={tmp}', '--com-convexa'],
            capture_output=True, text=True, cwd=ROOT
        )
        print(result.stdout)
        if result.stderr:
            for line in result.stderr.splitlines():
                if 'QFont' not in line:
                    print(f"  STDERR: {line}")

    os.unlink(tmp)
    print(f"  (arquivo temporário removido)")
    print()
