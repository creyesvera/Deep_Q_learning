import json
from pathlib import Path

def generar_dashboard_html(
    episode_states,
    template_path,
    output_path
):
    """
    episode_states: lista de dicts (Python)
    template_path: ruta al HTML plantilla
    output_path: ruta del HTML final
    """

    # Leer plantilla
    template_html = Path(template_path).read_text(encoding="utf-8")

    # Convertir lista Python → JSON válido para JS
    episode_states_json = json.dumps(
        episode_states,
        indent=2,
        ensure_ascii=False
    )

    # Reemplazar placeholder
    final_html = template_html.replace(
        "__EPISODE_STATES__",
        episode_states_json
    )

    # Guardar HTML final
    Path(output_path).write_text(final_html, encoding="utf-8")



episode_states = [
    {
        "idx": 0,
        "temp_in": 27.5,
        "temp_out": 31.2,
        "ac": False,
        "clases": 0,          # Laboratorio cerrado
        "thermal": 2,         # Muy calurosa
        "n_personas": 0,
        "ubicacion": 0,       # Perimetral
        "action": 0           # Prender ventilación
    },
    {
        "idx": 1,
        "temp_in": 26.8,
        "temp_out": 31.0,
        "ac": True,
        "clases": 1,          # Laboratorio abierto
        "thermal": 2,
        "n_personas": 5,
        "ubicacion": 1,       # Centro
        "action": 1           # Bajar temperatura
    },
    {
        "idx": 2,
        "temp_in": 25.9,
        "temp_out": 30.5,
        "ac": True,
        "clases": 1,
        "thermal": 1,         # Neutra
        "n_personas": 12,
        "ubicacion": 1,       # Centro
        "action": 2           # No hacer nada
    },
    {
        "idx": 3,
        "temp_in": 24.8,
        "temp_out": 29.9,
        "ac": True,
        "clases": 2,          # Clases prácticas
        "thermal": 1,
        "n_personas": 18,
        "ubicacion": 2,       # Zona ventanas
        "action": 2
    },
    {
        "idx": 4,
        "temp_in": 23.9,
        "temp_out": 29.4,
        "ac": True,
        "clases": 2,
        "thermal": 0,         # Muy fría
        "n_personas": 20,
        "ubicacion": 2,       # Zona ventanas
        "action": 3           # Subir temperatura
    },
    {
        "idx": 5,
        "temp_in": 24.2,
        "temp_out": 29.0,
        "ac": False,
        "clases": 2,
        "thermal": 1,
        "n_personas": 20,
        "ubicacion": 2,       # Zona ventanas
        "action": 4           # Apagar ventilación
    }
]
generar_dashboard_html(episode_states, "dashboard_template.html", "dashboard.html")