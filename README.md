# IPOP-MA-ES

## Requirements
* miniconda (or anaconda)
* coco benchmark framework for python (see installation at https://github.com/numbbo/coco/tree/master)

## Creating environment
1. Run:
```
conda env create -f environment.yml
```
2. Activate environment
```
conda activate amhe
```
3. Clone coco benchmark framework
```
git clone https://github.com/numbbo/coco.git
```
4. Install coco benchmark framework
```
cd coco
python do.py run-python
```
## Recreating results
Simply run:
```
python run_cmaes_experiments.py
python run_ipop_cmaes_experiments.py
python run_ipop_maes_experiments.py
```
and then process the results with the `process_results.ipynb` notebook.