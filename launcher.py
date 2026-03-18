import subprocess
import os



notebooks_to_run = '''
./notebooks/synthetic_SV_AE_compression_H4-4_L2.ipynb
./notebooks/synthetic_SV_AE_compression_H4-4_L3.ipynb
./notebooks/synthetic_SV_AE_compression_H8-4_L2.ipynb
./notebooks/synthetic_SV_AE_compression_H8-4_L3.ipynb
'''.split()



def run_notebook(nb):
  assert os.path.exists(nb), f'Path {nb} not found'
  subprocess.run(['sbatch', 'submit.sh', nb])

if __name__ == '__main__':
  print('Running', notebooks_to_run)
  for nb in notebooks_to_run:
    run_notebook(nb)
