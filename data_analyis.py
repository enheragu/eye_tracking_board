import os
import yaml
import pickle
import hashlib
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm

# Configuración global
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(CURRENT_FILE_PATH, 'output/plots/data_cache.pkl')
HASH_FILE = os.path.join(CURRENT_FILE_PATH, 'output/plots/data_hash.txt')

# Configuración de plots
PLOT_CONFIG = {
    'output_path': os.path.join(CURRENT_FILE_PATH, 'output/plots'),
    'image_path': os.path.join(CURRENT_FILE_PATH, 'media/TableroSinBordes.png'),
    'cmap': LinearSegmentedColormap.from_list('custom', ['blue', 'cyan', 'lime', 'yellow', 'red']),
    'participant_colors': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#a6cee3', '#b2df8a', '#fdbf6f', '#cab2d6', '#ffff99'
    ],
}

def load_background_image(img_path, attenuation_factor=0.3):
    """Carga y atenúa la imagen de fondo"""
    img = Image.open(img_path)
    width, height = img.size
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, int(255 * (1 - attenuation_factor))))
    img_attenuated = Image.alpha_composite(img, overlay)
    return np.array(img_attenuated.convert('RGB')), width, height

def calculate_data_hash(root_dir):
    """Calcula hash de los datos para validar caché"""
    hasher = hashlib.sha256()
    for dir_name in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            yaml_file = os.path.join(dir_path, f'data_{dir_name}.yaml')
            if os.path.exists(yaml_file):
                with open(yaml_file, 'rb') as f:
                    hasher.update(f.read())
    return hasher.hexdigest()

def load_participant_data(args):
    """Función para multiprocesamiento"""
    dir_name, root_dir = args
    dir_path = os.path.join(root_dir, dir_name)
    yaml_file = os.path.join(dir_path, f'data_{dir_name}.yaml')
    
    if not os.path.exists(yaml_file):
        return None
        
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
        return {
            'participant': dir_name,
            'data': data,
            'hash': calculate_data_hash(dir_path)
        }

def load_or_process_data(root_dir, force_reload=False):
    """Carga datos desde caché o procesa con multiprocesamiento"""
    current_hash = calculate_data_hash(root_dir)
    
    if not force_reload and os.path.exists(CACHE_FILE):
        with open(HASH_FILE, 'r') as f:
            saved_hash = f.read()
        if saved_hash == current_hash:
            with open(CACHE_FILE, 'rb') as f:
                print(f"Load data from cache file {CACHE_FILE}")
                return pickle.load(f)
    
    print(f"No cache found or data changed, processing data from scratch")
    dirs = [(d, root_dir) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(load_participant_data, dirs), total=len(dirs), desc='Processing participants'))
    
    all_data = [r for r in results if r is not None]
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(all_data, f)
    with open(HASH_FILE, 'w') as f:
        f.write(current_hash)
    
    return all_data

def plot_combined_data(all_data):
    """Genera gráficos combinados con validación estricta de trials"""
    bg_img, img_width, img_height = load_background_image(PLOT_CONFIG['image_path'])
    
    # Diccionario para agrupar trials compatibles
    trial_groups = {}
    
    # Primer paso: identificar trials únicos válidos
    for participant_data in all_data:
        participant = participant_data['participant']
        data = participant_data['data']
        
        for block_idx, trial_block in enumerate(data['trials_data']):
            print(f"Participant: {participant} - Block {block_idx}")
            for trial in trial_block:
                target_name = list(trial.keys())[0]
                trial_data = trial[target_name]
                
                required_keys = ['trial_id', 'sequence']
                if not all(k in trial_data for k in required_keys):
                    continue
                
                trial_key = f"{trial_data['trial_id']}_{target_name}"
                if (trial_data['trial_id'] < 0 
                    or 'missing_trial_error_' in trial_key 
                    or 'transition_error_' in trial_key):
                    continue

                print(f"    {trial.keys()} - {trial_data['trial_id']}")
                if block_idx not in trial_groups:
                    trial_groups[block_idx] = {}
                if trial_key not in trial_groups[block_idx]:
                    trial_groups[block_idx][trial_key] = {}
                trial_groups[block_idx][trial_key][participant] = trial_data['sequence']
    

    for block_idx, trials in trial_groups.items():
        print(f"Block {block_idx}:")
        for trial_key, participants_data in trials.items():
            print(f"  · Trial {trial_key}:")
            for participant_key, trial_data in participants_data.items():
                print(f"    - Participant {participant_key}")
    

    for block_idx, trials in tqdm(trial_groups.items(), desc="Generating plots per block"):
        num_trials = len(trials)
        num_cols = int(np.ceil(np.sqrt(num_trials)))  # Ajustar el número de columnas
        num_rows = int(np.ceil(num_trials / num_cols))  # Ajustar el número de filas
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))  # Ajustar tamaño

        # Si solo hay un subplot, axes no es una lista
        if num_trials == 1:
            axes = [axes]
        elif num_rows * num_cols > num_trials:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            for i in range(num_trials, num_rows * num_cols):
                fig.delaxes(axes[i])

        for ax in axes:
            ax.axis('off')
            
        trial_idx = 0
        for trial_key, participants_data in trials.items():
            trial_id, shape, color = trial_key.split('_')
            ax = axes[trial_idx]
            ax.imshow(bg_img, extent=[0, img_width, 0, img_height], origin='upper', interpolation='none')
            ax.set_xlim(0, img_width)
            ax.set_ylim(0, img_height)
            ax.set_aspect('equal')
            ax.axis('off')

            ax.set_title(f"Trial {trial_id}: {shape} {color}")

            for idx, (participant_key, trial_data) in enumerate(participants_data.items()):
                puntos = denormalize_coordinates([step['norm_board_coord'] for step in trial_data],
                                                 img_width, img_height)
                x = [p[0] for p in puntos]
                y = [img_height - p[1] for p in puntos]

                ax.plot(x, y, color=PLOT_CONFIG['participant_colors'][idx % len(PLOT_CONFIG['participant_colors'])],
                        label=f"{participant_key}")

            ax.legend()  # Agregar leyenda a cada subgráfica
            trial_idx += 1

        fig.suptitle(f"Block: {block_idx}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar diseño

        save_plot(fig, f"combined_block{block_idx}_all_trials")
        plt.close()


def denormalize_coordinates(norm_coords, img_width, img_height):
    """Convierte coordenadas normalizadas a píxeles en la imagen"""
    return [(x * img_width, y * img_height) for x, y in norm_coords]

def plot_single_trial(ax, trial_data, color, label=None):
    """Dibuja un trial individual en el axis proporcionado"""
    
    puntos = denormalize_coordinates(
        [step['norm_board_coord'] for step in trial_data],
        ax.img_width, 
        ax.img_height
    )
    x = [p[0] for p in puntos]
    y = [ax.img_height - p[1] for p in puntos]
    
    # colors = np.linspace(0, 1, len(puntos))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # lc = LineCollection(
    #     segments, 
    #     cmap=PLOT_CONFIG['cmap'],
    #     array=colors[:-1],
    #     linewidth=2,
    #     alpha=0.7,
    #     label=label
    # )
    # ax.add_collection(lc)
    # ax.scatter(x, y, c=colors, cmap=PLOT_CONFIG['cmap'], s=30, edgecolor='black', alpha=0.7)

    lc = LineCollection(
        segments, 
        colors=color,
        linewidth=2,
        alpha=0.7,
        label=label
    )
    ax.add_collection(lc)
    ax.scatter(x, y, c=color, s=30, edgecolor='black', alpha=0.7)

def plot_individual_data(all_data):
    """Genera gráficos individuales para cada participante"""
    bg_img, img_width, img_height = load_background_image(PLOT_CONFIG['image_path'])
    
    for participant_data in tqdm(all_data, desc='Generating individual plots'):
        participant = participant_data['participant']
        data = participant_data['data']
        
        for block_idx, trial_block in enumerate(data['trials_data']):
            for trial in trial_block:
                target_name = list(trial.keys())[0]
                trial_data = trial[target_name]
                
                if 'sequence' not in trial_data:
                    continue
                
                # Preparar figura
                fig, ax = prepare_plot(bg_img, img_width, img_height)
                ax.set_title(f"{target_name}\nParticipant: {participant} - Block: {block_idx} - Trial: {trial_data['trial_id']}")
                
                # Procesar y dibujar datos específicos
                norm_points = [step['norm_board_coord'] for step in trial_data['sequence']]
                puntos = denormalize_coordinates(norm_points, img_width, img_height)
                x = [p[0] for p in puntos]
                y = [img_height - p[1] for p in puntos]
                
                # Gradiente temporal
                colors = np.linspace(0, 1, len(puntos))
                segments = create_segments(x, y)
                
                # Dibujar elementos
                lc = LineCollection(
                    segments,
                    cmap=PLOT_CONFIG['cmap'],
                    array=colors[:-1],
                    linewidth=3,
                    alpha=0.8
                )
                ax.add_collection(lc)
                
                scatter = ax.scatter(x, y, c=colors, cmap=PLOT_CONFIG['cmap'], 
                                   s=50, edgecolor='black', zorder=2, alpha=0.8)
                
                # Añadir barra de color
                cbar = plt.colorbar(scatter, label='Time Trial')
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['T_init', 'T_end'])
                cbar.ax.yaxis.set_label_position('left')
                
                # Guardar
                save_plot(fig, f"{participant}_block{block_idx}_trial{trial_data['trial_id']}_{target_name}")
                plt.close()

def create_segments(x, y):
    """Helper para crear segmentos de línea"""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def prepare_plot(bg_img, img_width, img_height):
    """Prepara la figura base con la imagen de fondo"""
    fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=150)
    ax.imshow(bg_img, extent=[0, img_width, 0, img_height], origin='upper', interpolation='none')
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.img_width = img_width  # Almacenar dimensiones en el axis
    ax.img_height = img_height
    return fig, ax

def add_legend(ax):
    """Añade leyenda personalizada"""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc='upper right',
            frameon=True,
            facecolor='white',
            edgecolor='black',
            fontsize=8
        )

def save_plot(fig, filename):
    """Guarda el plot generado"""
    os.makedirs(PLOT_CONFIG['output_path'], exist_ok=True)
    fig.savefig(
        os.path.join(PLOT_CONFIG['output_path'], f"{filename}.svg"),
        bbox_inches='tight',
        pad_inches=0,
        dpi=300
    )
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', action='store_true', help='Forzar recarga de datos')
    parser.add_argument('--plot-type', choices=['individual', 'combined', 'both'], default='both')
    args = parser.parse_args()
    
    all_data = load_or_process_data(os.path.join(CURRENT_FILE_PATH, 'output/gaze'), force_reload=args.reload)
    
    if args.plot_type in ['individual', 'both']:
        print(f"Plot individual data")
        plot_individual_data(all_data)
    
    if args.plot_type in ['combined', 'both']:
        print(f"Plot combined data")
        plot_combined_data(all_data)
