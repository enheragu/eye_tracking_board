import os
from pathlib import Path
from src.utils import parseYaml

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
base_path = Path(f"{CURRENT_FILE_PATH}/output/gaze")
trial_config_path = Path(f"{CURRENT_FILE_PATH}/cfg/default_trials_config.yaml")
expected_files = [
    'data_{}.pkl',
    'trials_data_{}.csv',
    'data_{}.yaml',
    'result_log_{}.txt',
    'trials_data_{}_sequence.csv'
]
error_tags = ["missing_trial_error_", "transition_error_"]
ERROR_THRESHOLD = 2  # Tests with more than this N errors are reported

directories = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]

if not directories:
    print("❌ No se encontraron directorios numéricos.")
    exit()

missing_files_dirs = []
high_error_dirs = []

global_counts = {tag: 0 for tag in error_tags}

for dir_num in directories:
    dir_path = base_path / dir_num
    missing_files = []
    
    # Verify that has all needed files
    for template in expected_files:
        if not (dir_path / template.format(dir_num)).exists():
            missing_files.append(template.format(dir_num))
    
    if missing_files:
        missing_files_dirs.append(dir_num)
        print(f"\n📁 Directorio {dir_num}: ❌ Faltan {len(missing_files)} archivos")
        print("  - Archivos faltantes:", ", ".join(missing_files))
        continue
    
    # Verifyexecutions errors in those files
    dir_errors = {tag: 0 for tag in error_tags}
    yaml_file = dir_path / f'data_{dir_num}.yaml'
    
    yaml_content = parseYaml(yaml_file)
    if yaml_content:
        content_str = str(yaml_content).lower()
        for tag in error_tags:
            count = content_str.count(tag.lower())
            dir_errors[tag] = count
            global_counts[tag] += count
 
    
    total_errores = sum(dir_errors.values())
    if total_errores > 0:
        print(f"\n📁 Directorio {dir_num}:")
        for tag, count in dir_errors.items():
            if count > 0:
                print(f"  - {tag}: {count} ocurrencias")
        
        if total_errores > ERROR_THRESHOLD:
            high_error_dirs.append(dir_num)




print("\n" + "="*50)
print("🔍 LISTA DE DIRECTORIOS INCOMPLETOS:")
print('"' + '" "'.join(missing_files_dirs) + '"' if missing_files_dirs else "Ninguno")

print("\n🔍 LISTA DE DIRECTORIOS CON MÁS DE", ERROR_THRESHOLD, "ERRORES:")
print('"' + '" "'.join(high_error_dirs) + '"' if high_error_dirs else "Ninguno")

print("\n🔍 LISTA DE DIRECTORIOS CORRECTOS:")
correct_dirs = list(set(directories) - set(missing_files_dirs) - set(high_error_dirs))
print('"' + '" "'.join(correct_dirs) + '"' if correct_dirs else "Ninguno")

print("\n📊 RESUMEN GLOBAL:")
print(f"  - Directorios analizados: {len(directories)}")
print(f"  - Directorios incompletos: {len(missing_files_dirs)}")
print(f"  - Directorios con errores > {ERROR_THRESHOLD}: {len(high_error_dirs)}")
for tag, total in global_counts.items():
    print(f"  - Total {tag}: {total}")
