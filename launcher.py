import subprocess
import os



notebooks_to_run = '''
./notebooks/baseline_L3.ipynb
./notebooks/higher_beta_later.ipynb
./notebooks/L2_combination_beta_and_spectral.ipynb
./notebooks/L3_combination_beta_and_spectral.ipynb
./notebooks/L3_combination_lower_beta_and_higher_spectral.ipynb
./notebooks/L3_combination_lower_beta_and_spectral.ipynb
./notebooks/L3_nonVar_bet_spect.ipynb
./notebooks/nonVariational_L3_L10_spectral_50.ipynb
./notebooks/oscil_term_30.ipynb
./notebooks/oscil_term_50.ipynb
./notebooks/small_beta.ipynb
'''.split()



def run_notebook(nb):
  assert os.path.exists(nb), f'Path {nb} not found'
  subprocess.run(['sbatch', 'submit.sh', nb])

if __name__ == '__main__':
  print('Running', notebooks_to_run)
  for nb in notebooks_to_run:
    run_notebook(nb)
