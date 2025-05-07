import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(CURRENT_FILE_PATH, 'output/plots')
data_path = os.path.join(CURRENT_FILE_PATH, 'output/gaze')
image_path = os.path.join(CURRENT_FILE_PATH, 'media/TableroSinBordes.png')

def load_background_image(img_path, attenuation_factor=0.3):
    img = Image.open(img_path)
    width, height = img.size
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, int(255 * (1 - attenuation_factor))))
    img_attenuated = Image.alpha_composite(img, overlay)
    img_attenuated = img_attenuated.convert('RGB')
    return np.array(img_attenuated), width, height


def process_dirs(root_dir='.'):
    """Itera sobre los directorios y procesa los archivos YAML"""
    # Cargar imagen de fondo una sola vez
    bg_img, img_width, img_height = load_background_image(image_path)
    
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            yaml_file = os.path.join(dir_path, f'data_{dir_name}.yaml')
            print(f"Process participant: {dir_name}")
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    process_trials(data, dir_name, bg_img, img_width, img_height)
                    return

def process_trials(data, dir_name, bg_img, img_width, img_height):
    """Procesa todos los trials de un archivo YAML"""
    for block_index, trial_block in enumerate(data['trials_data']):
        for trial in trial_block:
            target_name = list(trial.keys())[0]
            trial_data = trial[target_name]

            if 'sequence' not in trial_data:
                print(f"[ERROR] Key 'sequence' not found in trial_data: {trial_data}")
                continue
            if 'trial_id' not in trial_data:
                print(f"[ERROR] Key 'trial_id' not found in trial_data: {trial_data}")
                continue
    
            print(f"\t· Processing test [{block_index}][{trial_data['trial_id']}]: {target_name}")
            plot_data(block_index, trial_data, target_name, dir_name, bg_img, img_width, img_height)

def denormalize_coordinates(norm_coords, img_width, img_height):
    """Convierte coordenadas normalizadas a píxeles en la imagen"""
    return [(x * img_width, y * img_height) for x, y in norm_coords]

def plot_data(block_index, trial_data, target_name, dir_name, bg_img, img_width, img_height):
    """Genera el gráfico con gradiente de color sobre la imagen de fondo"""
    
    # Desnormalizar coordenadas
    norm_points = [step['norm_board_coord'] for step in trial_data['sequence']]
    puntos = denormalize_coordinates(norm_points, img_width, img_height)
    x = [p[0] for p in puntos]
    y = [img_height - p[1] for p in puntos]  # Invertir eje Y para coincidir con matplotlib
    
    # Configurar figura
    fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
    ax.set_title(f"{target_name}\nParticipant: {dir_name} - Block: {block_index} - Trial: {trial_data['trial_id']}")
    ax.imshow(bg_img, extent=[0, img_width, 0, img_height], origin='upper', interpolation='none')
    
    # Crear gradiente de color
    colors = np.linspace(0, 1, len(puntos))
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'cyan', 'lime', 'yellow', 'red'])
    
    # Crear segmentos para el gradiente continuo
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=cmap, array=colors[:-1], linewidth=3, alpha=0.8)
    ax.add_collection(lc)
    
    scatter = ax.scatter(x, y, c=colors, cmap=cmap, s=50, edgecolor='black', zorder=2, alpha=0.8)
    
    # Ajustar límites y aspecto
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_aspect('equal')
    ax.axis('off')
    
    cbar = plt.colorbar(scatter, label='Time Trial')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['T_init', 'T_end'])
    cbar.ax.yaxis.set_label_position('left')
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label('Time Trial', fontsize=12)
    

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        f"{output_path}/{dir_name}_block{block_index}_trial{trial_data['trial_id']}_{target_name}.png",
        bbox_inches='tight',
        pad_inches=0,
        dpi=300
    )
    plt.close()

if __name__ == '__main__':
    process_dirs(data_path)
