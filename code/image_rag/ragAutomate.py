import pandas as pd
import shutil, os, subprocess
from ast import literal_eval

promptsDfPath = os.environ.get("RAGARTIST_PROMPTS_XLSX", "captions.xlsx")
API_key = os.environ["OPENAI_API_KEY"]

prompts = pd.read_excel(promptsDfPath,index_col = 'id')

for id in prompts.index:
  row = prompts.loc[id]
  prompt = row['Persian']
  english = row['English']
  print(f'Persian prompt: {prompt}','\n',f'English Translation: {english}')

  dest_folder = f"datasets/{row['English']}"
  os.makedirs(dest_folder, exist_ok = True)

  for j in range(len(literal_eval(row['Persian_Concepts']))):
    # Folder of Persian concept images
    source_folder = f"fromWeb/{english}_{j}"

    try:
        shutil.copytree(source_folder, dest_folder,dirs_exist_ok=True)
        print(f"Folder '{source_folder}' copied successfully to '{dest_folder}'.")
    except FileExistsError:
        print(f"Error: Destination folder '{dest_folder}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

  #AA#> Run RAG with OmniGen
  procStr = ['--prompt',  prompt, '--mode', 'generation','--out_name', english,
             '--dataset', english,'--omnigen_path', os.environ.get("OMNIGEN_DIR", "../OmniGen-main"),
             '--openai_api_key',  API_key
            ]

  ragProcess = subprocess.run([ "conda", "run", "-n", "ImageRAG",
                              "python", "imageRAG_OmniGen.py"] + procStr,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True # decode bytes → str
                          )

  print("STDOUT:\n", ragProcess.stdout)
  print("STDERR:\n", ragProcess.stderr)
