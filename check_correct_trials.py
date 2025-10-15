import os
import yaml

from concurrent.futures import ProcessPoolExecutor, as_completed

from tabulate import tabulate

from src.utils import parseYaml

# Configuración global
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_FILE_PATH, 'output/gaze')
trials_config_path = os.path.join(CURRENT_FILE_PATH, 'cfg/default_trials_config.yaml')
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

def table_to_html(table, headers,  output_file="output/check_trials_table.html"):
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

def main():
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
