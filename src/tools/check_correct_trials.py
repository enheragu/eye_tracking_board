import os
import sys
import yaml

from concurrent.futures import ProcessPoolExecutor, as_completed

from tabulate import tabulate

# Tools live in tools/, make the repo root importable (src package, entry points)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.core.version import __version__
from src.core.utils import parseYaml

# Configuración global
DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', f'/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData_v{__version__}')
DATA_PATH = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze')
trials_config_path = os.path.join(REPO_ROOT, 'cfg/default_trials_config.yaml')
yaml_file_name = "data_{}.yaml"
BLOCKS_CONFIG = parseYaml(trials_config_path)['test_block_list']


GREEN = "\033[92m"
RED = "\033[91m"
ORANGE = "\033[9202m"
RESET = "\033[0m"

mising_trial_tag = "missing_trial_error_" 
no_end_tag = "transition_error_no_end_"
no_init_tag = "transition_error_no_init_"
def get_participant_trials(participant_dir, participant_id):
    """Lee el archivo YAML de un participante y devuelve los trials presentes."""
    yaml_file = os.path.join(participant_dir, yaml_file_name.format(participant_id))
    if not os.path.exists(yaml_file):
        return None
    
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
        participant_id = data.get('participant_id', 'Unknown')
        trials_data = data.get('trials_data', [])
        
        # Crear un diccionario con los bloques y trials presentes
        print(f"📁 Processing Participant {participant_id}:")
        participant_trials = {}
        for block_id, trial_id in sorted(trials_data.keys()):
            trial = trials_data[(block_id, trial_id)]
            trial_name = list(trial.keys())[0]
            # trial_id = trial[trial_name].get('trial_id')
            participant_trials[(block_id, trial_id)] = {'trial_name': trial_name, 
                                                        'end_capture': trial[trial_name]['end_capture'], 
                                                        'init_capture': None if 'init_capture' not in trial[trial_name] else trial[trial_name]['init_capture']}
            print(f"  · Block {block_id}: {trial_id = }; {trial_name = }")
        
        print(f"   Processed participant {participant_id}: {len(participant_trials)} trials found.")#:\n{participant_trials}")
        return participant_id.strip(), participant_trials

def generate_presence_table(participants_data):
    """Genera una tabla de presencia de trials para todos los participantes."""
    table = []
    headers = ["Trial", "Name"] + list(sorted(participants_data.keys())) + ['N. Ok']
    
    no_end_timestampts = {}
    for block_id, block_data in BLOCKS_CONFIG:
        for trial_id, trial_name in block_data:            
            row = []
            row.append(f"[{block_id},{trial_id}] ")
            row.append(f"{trial_name}")
            
            test_ok = 0
            for participant_id in sorted(participants_data.keys()):
                trials = participants_data[participant_id]
                if (block_id, trial_id) in trials:
                    current_trial = trials[(block_id, trial_id)]
                    current_trial_name = current_trial['trial_name']
                    if current_trial_name == trial_name:
                        row.append(GREEN+"Yes"+RESET) #("✅")
                        test_ok +=1
                    elif mising_trial_tag in current_trial_name and current_trial_name.replace(mising_trial_tag, "") == trial_name:
                        row.append(RED+"Miss"+RESET) #("✅")
                    elif no_end_tag in current_trial_name and current_trial_name.replace(no_end_tag, "") == trial_name:
                        row.append(RED+"No-end"+RESET) #("✅")
                        if not participant_id in no_end_timestampts:
                            no_end_timestampts[participant_id] = {}
                        if not block_id in no_end_timestampts[participant_id]:
                            no_end_timestampts[participant_id][block_id] = {}
                        no_end_timestampts[participant_id][block_id][trial_id] = {'name': current_trial_name.replace(no_end_tag, ""),
                                                                                  'end_capture': current_trial['end_capture'],
                                                                                  'init_capture': current_trial['init_capture']}
                    elif no_init_tag in current_trial_name and current_trial_name.replace(no_init_tag, "") == trial_name:
                        row.append(RED+"No-init"+RESET) #("✅")
                        if not participant_id in no_end_timestampts:
                            no_end_timestampts[participant_id] = {}
                        if not block_id in no_end_timestampts[participant_id]:
                            no_end_timestampts[participant_id][block_id] = {}
                        no_end_timestampts[participant_id][block_id][trial_id] = {'name': current_trial_name.replace(no_init_tag, ""),
                                                                                  'end_capture': current_trial['end_capture'],
                                                                                  'init_capture': current_trial['init_capture']}
                    else:
                        row.append(RED+"No"+RESET) #("❌")  # Presente
                else:
                    row.append(RED+"--"+RESET) #("❌")  # Presente    
            row.append(f"{test_ok}/{len(participants_data.keys())}")       
            table.append(row)
    
    for participant, participant_data in no_end_timestampts.items():
        print(f"Participant {participant}:")
        for block_id, bock_data in participant_data.items():
            for trial_id, trial_data in bock_data.items():
                print(f"  · [{block_id}][{trial_id}] {trial_data = }")

    return headers, table

def table_to_html(table, headers,  output_file=os.path.join(DEFAULT_OUTPUT_ROOT, "check_trials_table.html")):
    ansi_to_html = {
        "\033[92m": '<span class="green">',
        "\033[91m": '<span class="red">',
        "\033[9202m": '<span class="orange">',
        "\033[0m": '</span>'
    }

    def reemplazar_ansi(texto):
        for ansi, html in ansi_to_html.items():
            texto = texto.replace(ansi, html)
        return texto

    table_html = [
        [reemplazar_ansi(str(cell)) for cell in row]
        for row in table
    ]

    html_table = tabulate(table_html, headers=headers, tablefmt="unsafehtml")
    # Find header row
    start = html_table.find("<tr>")
    end = html_table.find("</tr>", start) + len("</tr>")
    header_row = html_table[start:end]

    # Insert header row also at the end as a footer :)
    html_table_with_footer = html_table.replace("</table>", f"{header_row}\n</table>")

    css = """
    <style>
    .green { color: #28a745; font-weight: bold; }
    .red { color: #dc3545; font-weight: bold; }
    .orange { color: orange; font-weight: bold; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #333; padding: 8px; text-align: center; }
    th { background-color: #f2f2f2; }
    </style>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html>\n<html>\n<head>{css}</head>\n<body>\n{html_table_with_footer}\n</body>\n</html>")
    


def process_dir(dir_name):
    dir_path = os.path.join(DATA_PATH, dir_name)
    if os.path.isdir(dir_path):
        result = get_participant_trials(dir_path, dir_name)
        if result:
            participant_id, trials_present = result
            return participant_id, trials_present
    return None

def check_config_sanity():
    """Capa de seguridad (estática): revisa las configs de excepción por participante frente al
    default y avisa del patrón peligroso -- un trial REAL (trial_id>=0 en el default) convertido
    en un [-1] a descartar en la excepción, que silenciosamente PIERDE ese trial. Es la clase de
    error de 001 (red_hexagon t4 -> -1) y de 035/055 (yellow_circle t0 -> -1). Distingue el `-1`
    legítimo de re-presentación (Vane/064: añaden un -1 SIN quitar el trial real) del erróneo."""
    exc_dir = os.path.join(REPO_ROOT, 'cfg', 'trials_config_exceptions')
    if not os.path.isdir(exc_dir):
        return
    default = {b[0]: b[1] for b in parseYaml(trials_config_path)['test_block_list']}
    print("\n=== CONFIG SAFETY: excepciones de trials vs default ===")
    n_susp = 0
    for fname in sorted(os.listdir(exc_dir)):
        if not fname.endswith('_trials_config.yaml'):
            continue
        pid = fname.replace('_trials_config.yaml', '')
        path = os.path.join(exc_dir, fname)
        exc = {b[0]: b[1] for b in parseYaml(path)['test_block_list']}
        # ¿hay comentario EXPLICATIVO (más allá del boilerplate)? Un -1 que descarta un trial
        # real es legítimo si está documentado (035/055: el participante no entendió la tarea)
        # y SOSPECHOSO si es mudo (001 lo era). parseYaml quita comentarios, así que se lee crudo.
        boiler = ('All 6 blocks', 'Includes [block_id')
        documented = any(l.strip().startswith('#') and not any(b in l for b in boiler)
                         for l in open(path).read().splitlines())
        flags = []
        for blk in sorted(set(default) | set(exc)):
            d, e = default.get(blk, []), exc.get(blk, [])
            if d == e:
                continue
            d_real = {n for t, n in d if t >= 0}
            e_real = {n for t, n in e if t >= 0}
            e_minus1 = {n for t, n in e if t == -1}
            # trial real del default que en la excepción ya NO es real y aparece como -1
            suspicious = sorted((d_real - e_real) & e_minus1)
            if suspicious:
                flags.append((blk, suspicious))
        if flags:
            mudo = not documented
            if mudo:
                n_susp += 1
            tag = f"{RED}SIN COMENTARIO -> SOSPECHOSO{RESET}" if mudo else "documentado (exclusión deliberada, OK)"
            print(f"  [{pid}] {tag}")
            for blk, susp in flags:
                print(f"     bloque {blk}: trial REAL -> -1 {susp}")
        elif exc == default:
            print(f"  [{pid}] idéntico al default (excepción innecesaria; se puede borrar)")
    if not n_susp:
        print("  OK: ningún -1 que descarte un trial real está sin documentar")
    print()


def main():
    check_config_sanity()
    participants_data = {}
    with ProcessPoolExecutor(max_workers=6) as executor:  # Ajusta el número de workers según tu CPU
        futures = []
        for dir_name in os.listdir(DATA_PATH):
            futures.append(executor.submit(process_dir, dir_name))

        for future in as_completed(futures):
            result = future.result()
            if result:
                participant_id, trials_present = result
                participants_data[participant_id] = trials_present
    
    # print(participants_data)
    headers, table = generate_presence_table(participants_data)
    
    table_to_html(table, headers)
    print(tabulate(table,
                   headers=headers,
                   tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
