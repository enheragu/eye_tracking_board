<h1>Data extraction from visual search experiments in natural environments</h1>

## Table of contents
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Code description](#code-description)
- [Installation And Usage](#installation-and-usage)
- [Contributions and Dissemination](#contributions-and-dissemination)
- [Contributors and Contact Information](#contributors-and-contact-information)

## Introduction

The task consists on finding one of the following targets, that are shown to the participant in a panel in the board below.

<div align="center"><figure>
  <img src="./media/TableroSinBordes.png" width="40%" />
  <figcaption>Figure 1. Rendered image of the board with the different pieces on it.</figcaption>
</figure></div>

<div align="center"><figure>
<div style="display: flex; gap: 30px;">
  <img src="./media/documentation/targets_carousel.gif" width="30%" />
  <img src="./media/documentation/targets_carousel_2.gif" width="30%" />
  <img src="./media/documentation/targets_carousel_3.gif" width="30%" />
</div>  <figcaption>Figure 2. Panels with target to look for that is shown to the participants.</figcaption>
</figure></div>

The whole experiment is composed by six blocks of ten trials each.



|           | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Trial 6 | Trial 7 | Trial 8 | Trial 9 | Trial 10 |
|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|
| Block 1  | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![rojo hexágono](media/print_material/Estímulos_Imprimir/3.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![rojo triángulo](media/print_material/Estímulos_Imprimir/5.png) | ![amarillo hexágono](media/print_material/Estímulos_Imprimir/8.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) | ![verde hexágono](media/print_material/Estímulos_Imprimir/9.png) | ![amarillo triángulo](media/print_material/Estímulos_Imprimir/6.png) | ![rojo círculo](media/print_material/Estímulos_Imprimir/4.png) |
| Block 2  | ![rojo hexágono](media/print_material/Estímulos_Imprimir/3.png) | ![amarillo triángulo](media/print_material/Estímulos_Imprimir/6.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![rojo círculo](media/print_material/Estímulos_Imprimir/4.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) | ![amarillo hexágono](media/print_material/Estímulos_Imprimir/8.png) | ![rojo triángulo](media/print_material/Estímulos_Imprimir/5.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) |
| Block 3  | ![rojo círculo](media/print_material/Estímulos_Imprimir/4.png) | ![amarillo triángulo](media/print_material/Estímulos_Imprimir/6.png) | ![verde hexágono](media/print_material/Estímulos_Imprimir/9.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) | ![amarillo hexágono](media/print_material/Estímulos_Imprimir/8.png) | ![rojo triángulo](media/print_material/Estímulos_Imprimir/5.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) |
| Block 4  | ![amarillo hexágono](media/print_material/Estímulos_Imprimir/8.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![rojo triángulo](media/print_material/Estímulos_Imprimir/5.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![rojo hexágono](media/print_material/Estímulos_Imprimir/3.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![amarillo triángulo](media/print_material/Estímulos_Imprimir/6.png) | ![rojo círculo](media/print_material/Estímulos_Imprimir/4.png) |
| Block 5  | ![rojo círculo](media/print_material/Estímulos_Imprimir/4.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) | ![verde hexágono](media/print_material/Estímulos_Imprimir/9.png) | ![amarillo triángulo](media/print_material/Estímulos_Imprimir/6.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![rojo hexágono](media/print_material/Estímulos_Imprimir/3.png) | ![rojo triángulo](media/print_material/Estímulos_Imprimir/5.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) |
| Block 6  | ![verde hexágono](media/print_material/Estímulos_Imprimir/9.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![amarillo hexágono](media/print_material/Estímulos_Imprimir/8.png) | ![rojo círculo](media/print_material/Estímulos_Imprimir/4.png) | ![azul triángulo](media/print_material/Estímulos_Imprimir/2.png) | ![azul círculo](media/print_material/Estímulos_Imprimir/1.png) | ![verde triángulo](media/print_material/Estímulos_Imprimir/10.png) | ![amarillo círculo](media/print_material/Estímulos_Imprimir/7.png) | ![rojo triángulo](media/print_material/Estímulos_Imprimir/5.png) | ![rojo hexágono](media/print_material/Estímulos_Imprimir/3.png) |


## Code description

## Installation And Usage

It is recommended to install the setup into a virtual environment. Create one and activate it with the following command (or just ignore these an execute without venv):

```sh
    python3 -m venv path_venv
    source path_venv/bin/activate
```

The environment can be deactivated as follows:
```sh
    deactivate
```

Clone the repository in a given location and install its requirementes with the following command, executed from the root folder of the repository. You can check the requirements file to check the libraries that will be installed into your system.

```sh
    pip3 install -r requirements
```


## Contributions and Dissemination

Below is a summary of the key presentations and publications associated with the code and information contained in this repository:

- `[Poster presentation]` **Laura Cepero, Enrique Heredia-Aguado, Lidia Sobrino, Laura Cantero, María García de Viedma, Victoria Plaza, María Pilar Aivar.**  
  *XIV Congress of the Spanish Society for Experimental Psychology (SEPEX)* (Almería, España. Octubre 2024). [(Abstracts Book ](https://www.sepex24.com/es/_files/ugd/5182f6_b112e823d83243949557d760e3eb534c.pdf?index=true)   
  [**Bringing Visual Search to Life: how we find colored objects when performing a natural task.**](https://www.researchgate.net/publication/395732728_Bringing_Visual_Search_to_Life_how_we_find_colored_objects_when_performing_a_natural_task)
  

- `[Conference Talk]` **Enrique Heredia-Aguado, Laura Cepero, Luis Miguel Jiménez, Victoria Plaza, María Pilar Aivar.**  
  *V Congreso Anual de Estudiantes de Doctorado* (Elche, España. Febrero 2025).  [(Abstracts Book - TBC)]()   
  [**Colaboración interdisciplinar en estudios de doctorado: Estudiando el proceso de búsqueda visual en personas a través de la visión por computador.**](https://www.researchgate.net/publication/390364612_Colaboracion_interdisciplinar_en_estudios_de_doctorado_Estudiando_el_proceso_de_busqueda_visual_en_personas_a_traves_de_la_vision_por_computador)

- `[Conference Talk]` **Laura Cepero, Enrique Heredia-Aguado, Victoria Plaza, María Pilar Aivar.**  
  *Reunión Científica sobre Atención (RECA14)* (Madrid, España. Abril 2025). [(Abstracts Book - TBC)]()   
  [**Visual Search Strategies: Comparing Screen-Based and Real-World Contexts.**](https://www.researchgate.net/publication/396733528_Visual_Search_Strategies_Comparing_Screen-Based_and_Real-World_Contexts)



## Contributors and Contact Information