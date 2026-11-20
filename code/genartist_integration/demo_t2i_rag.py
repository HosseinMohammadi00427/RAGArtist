from PIL import Image
import os,shutil
import os.path as osp
import torch
#import diffusers
#import cv2
import gc
import json
from openai import OpenAI
import requests
import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def command_parse(commands, text, text_bg, dir='inputs'):
    args = []
    generation_arg = None
    for i in range(len(commands)):
        command = commands[i]
        if command['tool'] == 'edit':
            k = len(args)
            if 'box' in command:
                if 'intbox' in command:
                    bb = command['box']
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], 'box': command['box']} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg,
                    "output": osp.join(dir, str(k+2)+'.png'), "output_mask": osp.join(dir, str(k+2)+'_mask.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["edit"], "layout": osp.join(dir, str(k+1)+'_mask.png') } }
            args.append(arg)

            arg = {"tool": "replace_anydoor", "output": osp.join(dir, str(k+3)+'.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+2)+'.png'),
                             "object_mask": osp.join(dir, str(k+2)+'_mask.png'), "mask": osp.join(dir, str(k+1)+'_mask.png'),  } }
            args.append(arg)
        elif command['tool'] == 'move':
            if 'intbox' in command:
                bb = command['box']
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+3)+'.png'),
                "input": {"image": osp.join(dir, str(k+2)+'.png'), "object": osp.join(dir, str(k)+'.png'),
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command["box"]  } }
            args.append(arg)
        elif command['tool'] == 'addition':
            k = len(args)
            if 'intbox' in command:
                bb = command['box']
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg,
                    "output": osp.join(dir, str(k+1)+'.png'), "output_mask": osp.join(dir, str(k+1)+'_mask.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["input"], "layout": command["box"] } }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+2)+'.png'),
                "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+1)+'.png'),
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command["box"]  } }
            args.append(arg)
        elif command['tool'] == 'remove':
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], "mask_threshold": command.get("mask_thr", 0.0)} }
            if 'box' in command:
                if 'intbox' in command:
                    bb = command['box']
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg['input']['box'] = command['box']
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)
        elif command['tool'] == 'instruction':
            k = len(args)
            arg = {"tool": "instruction", "output": osp.join(dir, str(k+1)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)
        elif command['tool'] == 'edit_attribute':
            k = len(args)
            if 'box' in command:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], 'box': command['box']} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "attribute_diffedit",
                    "output": osp.join(dir, str(k+2)+'.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["input"], "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "attr": command["text"] } }
            args.append(arg)

        elif command['tool'] in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD', 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
            generation_arg = command
            generation_arg["output"] = osp.join(dir, '0.png')

    k = len(args)
    arg = {"tool": "superresolution_SDXL", "input": {"image": osp.join(dir, str(k)+'.png')}, "output": osp.join(dir, str(k+1)+'.png')}
    args.append(arg)
    if generation_arg is not None:
        args = [generation_arg] + args
    return args



if __name__ == '__main__':

    api_key = os.environ["OPENAI_API_KEY"]

    mainDir = os.environ.get("RAGARTIST_DATA_DIR", ".")
    origDir = os.path.join(os.environ.get("RAGARTIST_INPUT_DIR", os.path.join(mainDir, "inputs")), "")
    files = sorted(os.listdir(origDir))
    files = [fi for fi in files if fi.endswith('.png')]
    for filePath in files:
        try:
            prompt = filePath.split('_gs')[0]
            txtPath = prompt + '.txt'
            #AA#> For Persian Prompt
            #with open(origDir + txtPath,'r') as f: prompt = f.readline().strip('\n')

            print(f'\nProcessing image {prompt}')
            outDir = f'{mainDir}/Results/RAGArtist/{prompt}'
            for path in ['inputs',outDir]:
                os.makedirs(path, exist_ok= True)

            rag_path = f'inputs/0'+'.png'
            shutil.copy(origDir+filePath, rag_path)
            shutil.copy(origDir+txtPath, rag_path[:-3]+'txt')
            shutil.copytree("inputs/", outDir, dirs_exist_ok=True)
            print(f'PNG and TXT files for {prompt} are moved to results/{prompt}','\n')

            # GenArtist expects images to be of size (512,512)
            img_resized = Image.open(rag_path).resize((512, 512), Image.LANCZOS)
            img_resized.save(rag_path)

            url = "https://api.openai.com/v1/chat/completions"
            client = OpenAI(api_key=api_key)

            headers = {
                        'Accept': 'application/json',
                        'Authorization': f'Bearer {api_key}',
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }


            ## Detection
            seq_args = [{'tool':'detection', 'input':{'image':rag_path, 'text':'TBG'}}]
            json.dump(seq_args[-1], open('input.json','w'))
            print('Running Object Detection...')
            os.system('python agent_tool_aux.py --json_out True')


            ## verification and self-correction

            with open('prompts/correction.txt', 'r', encoding='utf-8') as f:
                template=f.readlines()
            user_textprompt = f"Caption: In Iran, {prompt}\n"
            detection_results = json.load(open('input_detection.json'))
            user_textprompt += 'I can give you the position of all objects in the image: ' + str(detection_results) + '\n'
            user_textprompt += 'You can use these as a reference for generating the bounding box position of objects'
            user_textprompt += """Please only output the editing operations through dict, do not output other analysis process.
                                    Return the result with only plain text, do not use any markdown or other style.
                                    All characters must be in English."""

            textprompt= f"{' '.join(template)} \n {user_textprompt}"
            image_path = rag_path
            base64_image = encode_image(image_path)
            payload = json.dumps({
                "model": "gpt-5",
                "messages": [
                    {
                        "role": "user",
                        "content": [{'type':'text', 'text':textprompt}, {'type':'image_url', 'image_url':{"url":  f"data:image/jpeg;base64,{base64_image}"} }]
                    }
                ]
                })
            print('waiting for GPT-5 response in verification and self-correction')
            response = requests.request("POST", url, headers=headers, data=payload)
            obj = response.json()
            correction_text = obj['choices'][0]['message']['content']
            print(correction_text)
            json.dump(correction_text, open('correction_text.json','w'))

            try:
                correction_text2 = '[' + correction_text.replace('}','},') + ']'
                print(correction_text2)
                correction_text = eval(correction_text2)
            except:
                correction_text = eval(correction_text)


            commands = correction_text if type(correction_text) is list else [correction_text]
            seq_args = command_parse(commands, prompt, '')


            for i in range(0, len(seq_args)):
                print('--'*50,'\n',i, seq_args[i])
                json.dump(seq_args[i], open('input.json','w'))
                if seq_args[i]['tool'] in ["object_addition_anydoor", "segmentation"]:
                    # main_aux(seq_args[i])
                    os.system('python agent_tool_aux.py --json_out True')
                elif seq_args[i]['tool'] in ["addition_anydoor", "replace_anydoor", "remove", "instruction", "attribute_diffedit"]:
                    # main_edit(seq_args[i])
                    os.system('python agent_tool_edit.py --json_out True')
                elif seq_args[i]['tool'] in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD', 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
                    # main_generate(seq_args[i])
                    os.system('python agent_tool_generate.py --json_out True')


            shutil.copytree("inputs/", outDir, dirs_exist_ok=True)
            shutil.rmtree("inputs/")
            json.dump(correction_text, open(f'{outDir}/correction_text.json','w'))
            json.dump(detection_results, open(f'{outDir}/detection_results.json','w'))
            print(f'Output files are saved at {outDir}')

        except Exception as e:
            print('***\n',e,'***\n')
