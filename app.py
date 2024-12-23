import re
import gradio as gr
import pandas as pd
import torch
import warnings
from sklearn.metrics.pairwise import cosine_similarity

from model import MimicTransformer
from utils import (
    load_rule, get_attribution, get_diseases, get_drg_link, 
    get_icd_annotations, visualize_attn, clean_text
)
from transformers import AutoTokenizer, AutoModel, set_seed, pipeline

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(0)
set_seed(34)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Model configuration
model_dict = {
    "mimic": {
        "model_path": 'checkpoint_0_9113.bin',
        "model_url": r"E:\\transformers\\models--emilyalsentzer--Bio_ClinicalBERT\\snapshots\\model",
    },
    "similarity": {
        "model_path": r'E:\\transformers\\models--kamalkraj--BioSimCSE-BioLinkBERT-BASE\\snapshots\\model',
        "embedding_file_path": "embeddings_cpu.pt",
        "csv_file_path": "cleaned_patients.csv",
    },
    "ner": {
        "model_url": r"E:\\transformers\\models--alvaroalon2--biobert_diseases_ner\\snapshots\\model",
    }
}

# Model initialization
print("*" * 20, "MODEL_INIT", "*" * 20)

# Initialize Mimic model
print("Mimic Init")
mimic = MimicTransformer(
    tokenizer_name=model_dict["mimic"]["model_url"], 
    cutoff=512, 
    model_path=model_dict["mimic"]["model_path"]
)
mimic_tokenizer = mimic.tokenizer
mimic.eval()

# Initialize similarity model
print("Similarity Init")
similarity_tokenizer = AutoTokenizer.from_pretrained(model_dict["similarity"]["model_path"])
similarity_model = AutoModel.from_pretrained(model_dict["similarity"]["model_path"])
similarity_model.eval()

# Initialize NER pipeline
print("NER Init")
pipe = pipeline("token-classification", model=model_dict["ner"]["model_url"])

# Load related data
related_tensor = torch.load(model_dict["similarity"]["embedding_file_path"])
all_summaries = pd.read_csv(model_dict["similarity"]["csv_file_path"])["patient"].to_list()

print("*" * 20, "END_MODEL_INIT", "*" * 20)

# Load DRG rules
rule_df, drg2idx, i2d, d2mdc, d2w = load_rule('MSDRG_RULE13.csv')

def run(text, related_discharges=False):
    # Reinitialize seeds for reproducibility
    torch.manual_seed(0)
    set_seed(34)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Process text with Mimic model
    def get_model_results(text):
        text = clean_text(text)
        inputs = mimic_tokenizer(
            text, return_tensors='pt', padding='max_length', max_length=512, truncation=True
        )
        with torch.no_grad():
            outputs = mimic(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                drg_labels=None
            )
        attribution, reconstructed_text = get_attribution(
            text=text, tokenizer=mimic_tokenizer, model_outputs=outputs, inputs=inputs, k=10
        )
        logits = outputs[0][0]
        out = logits.detach().cpu()[0]
        drg_code = i2d[out.argmax().item()]
        prob = torch.nn.functional.softmax(out).max()
        print(out.topk(5))
        return {
            'class': drg_code,
            'prob': prob,
            'attn': attribution,
            'tokens': reconstructed_text,
            'logits': logits
        }

    model_results = get_model_results(text=text)
    drg_code = model_results['class']

    # Extract diseases using NER pipeline
    diseases = get_diseases(text=text, pipe=pipe)
    model_results['diseases'] = diseases

    # Retrieve DRG link and annotations
    drg_link = get_drg_link(drg_code=drg_code)
    icd_results = get_icd_annotations(text=text)

    # Retrieve DRG description
    row = rule_df[rule_df['DRG_CODE'] == drg_code]
    drg_description = row['DESCRIPTION'].values[0]
    model_results.update({
        'class_dsc': drg_description,
        'drg_link': drg_link,
        'icd_results': icd_results
    })

    def find_related_summaries(text, top_k=5):
        # 将输入文本编码到模型
        inputs = similarity_tokenizer(
            text, padding='max_length', truncation=True, return_tensors='pt', max_length=512
        )

        # 获取模型输出
        with torch.no_grad():
            outputs = similarity_model(**inputs)

        # 获取 last_hidden_state 的平均值作为句子嵌入
        hidden_states = outputs.last_hidden_state
        query_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()  # 在tokens维度上取均值

        # 计算输入文本与所有摘要的余弦相似度
        scores = cosine_similarity([query_embedding], related_tensor).flatten()

        # 找到最高相似度的前 top_k 条
        topk_indices = scores.argsort()[-top_k:][::-1]  # 获取得分最高的索引，降序排列
        topk_scores = scores[topk_indices]
        print(topk_indices, topk_scores)

        # 生成结果列表
        summary_score_list = []
        for idx, score in zip(topk_indices, topk_scores):
            corresp_summary = all_summaries[idx]
            summary_score_list.append([round(score, 2), corresp_summary])

        return summary_score_list

    related_summaries = find_related_summaries(text=text)

    # Return results
    if related_discharges:
        return visualize_attn(model_results=model_results)

    return (
        visualize_attn(model_results=model_results),
        gr.update(value=related_summaries, visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True)
    )

def run_related():
    global related_chosen
    attn_list = []
    clr_bts = []
    correct_drg_list = []
    correct_salient_list = []
    for related in related_chosen:
        text = related[0]
        attn_html = run(text=text, related_discharges=True)
        attn_list.append(gr.HTML.update(value=attn_html))
        clr_bts.append(gr.ClearButton.update(visible=True))
        correct_drg_list.append(gr.Textbox.update(visible=True))
        correct_salient_list.append(gr.Textbox.update(visible=True))
    if len(attn_list) != 3:
        # find difference
        diff = 3 - len(attn_list)
        for i in range(diff):
            attn_list.append(gr.HTML.update(value=''))
            clr_bts.append(gr.ClearButton.update(visible=False))
            correct_drg_list.append(gr.Textbox.update(visible=False))
            correct_salient_list.append(gr.Textbox.update(visible=False))
    return attn_list + clr_bts + correct_drg_list + correct_salient_list

def load_example(example_id):
    global related_summaries
    global related_chosen
    sample = related_summaries[example_id][0]
    cleaned_sample = sample.split('% Similarity Rate for the following Discharge Summary:\n\n')[1:]
    related_chosen.append(cleaned_sample)
    return prettify_text(related_chosen)
    # return related_chosen

def load_df_example(df, event: gr.SelectData):
    global related_chosen
    discharge_summary = event.value
    related_chosen.append([discharge_summary])
    return prettify_text(related_chosen)

def save_results(text):
    return gr.Textbox.update(value='Thank you for your input!')

def prettify_text(nested_list):
    string = ''
    for li in nested_list:
        # 检查 li[0] 是否是字符串
        if isinstance(li[0], str):
            striped = re.sub(' +', ' ', li[0]).strip()
        else:
            striped = str(li[0])  # 强制转换为字符串
        delimiters = 99 * '='
        string += f'{striped}\n{delimiters}\n'
    return string.strip()


def remove_most_recent():
    global related_chosen
    related_chosen = related_chosen[:-1]
    if len(related_chosen) == 0:
        return ''
    return prettify_text(related_chosen)

def clr_btn():
    return gr.ClearButton.update(visible=False), gr.Textbox.update(visible=False), gr.Textbox.update(visible=False)


# default DRG summary examples
ex1 = """HEAD CT:  Head CT showed no intracranial hemorrhage or mass effect, but old infarction consistent with past medical history."""
ex2 = """Radiologic studies also included a chest CT, which confirmed cavitary lesions in the left lung apex consistent with infectious tuberculosis. This also moderate-sized left pleural effusion."""
ex3 = """We have discharged Mrs Smith on regular oral Furosemide (40mg OD) and we have requested an outpatient ultrasound of her renal tract which will be performed in the next few weeks. We will review Mrs Smith in the Cardiology Outpatient Clinic in 6 weeks time."""
ex4 = """Blood tests revealed a raised BNP. An ECG showed evidence of left-ventricular hypertrophy and echocardiography revealed grossly impaired ventricular function (ejection fraction 35%). A chest X-ray demonstrated bilateral pleural effusions, with evidence of upper lobe diversion."""
ex5 = """Mrs Smith presented to A&E with worsening shortness of breath and ankle swelling. On arrival, she was tachypnoeic and hypoxic (oxygen saturation 82% on air). Clinical examination revealed reduced breath sounds and dullness to percussion in both lung bases. There was also a significant degree of lower limb oedema extending up to the mid-thigh bilaterally."""
examples = [ex1, ex2, ex3, ex4, ex5]
related_summaries = [[ex1]]
related_chosen = []
related_attn = []
related_clr_bts = []
correct_drg_text_list = []
correct_salient_words_list = []

def main():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # DRGCoder
        This interface outlines DRGCoder, an explainable clinical coding for the early prediction of diagnostic-related groups (DRGs). Please note all summaries will be truncated to 512 words if longer. 
        """)
        with gr.Row() as row:
            input = gr.Textbox(
                label="Input Discharge Summary Here", placeholder='sample discharge summary',
                text_align='left', interactive=True
            )
        with gr.Row() as row:
            gr.Examples(examples, [input])
        with gr.Row() as row:
            btn = gr.Button(value="Submit")
        with gr.Row() as row:
            attn_viz = gr.HTML() 
        with gr.Row() as row:
            with gr.Column() as col:
                correct_drg_text = gr.Textbox(visible=False, label="Input Correct DRG", interactive=True)
                correct_drg_text.submit(save_results, inputs=correct_drg_text, outputs=correct_drg_text)            
            with gr.Column() as col:
                salient_words_box = gr.Textbox(visible=False, label="Input Salient Words (comma separated)", interactive=True)
                salient_words_box.submit(save_results, inputs=salient_words_box, outputs=salient_words_box)
            attn_clr_btn = gr.ClearButton(value='Remove DRG Results', visible=False, components=[attn_viz]) 
            attn_clr_btn.click(clr_btn, outputs=[attn_clr_btn, correct_drg_text, salient_words_box])        
        
        
        print("row 1")
        # related row 1
        with gr.Row() as row:
            attn_1 = gr.HTML()
            related_attn.append(attn_1)
        with gr.Row() as row:
            with gr.Column() as col:
                correct_drg_text_1 = gr.Textbox(visible=False, label="Input Correct DRG", interactive=True)
                correct_drg_text_1.submit(save_results, inputs=correct_drg_text_1, outputs=correct_drg_text_1)
                correct_drg_text_list.append(correct_drg_text_1)            
            with gr.Column() as col:
                salient_words_box_1 = gr.Textbox(visible=False, label="Input Salient Words (comma separated)", interactive=True)
                salient_words_box_1.submit(save_results, inputs=salient_words_box_1, outputs=salient_words_box_1)
                correct_salient_words_list.append(salient_words_box_1)
            attn_clr_1 = gr.ClearButton(value='Remove DRG Results', visible=False, components=[attn_1])    
            related_clr_bts.append(attn_clr_1)
            attn_clr_1.click(clr_btn, outputs=[attn_clr_1, correct_drg_text_1, salient_words_box_1]) 

        print("row 2")
        # related row 2
        with gr.Row() as row:
            attn_2 = gr.HTML()
            related_attn.append(attn_2)
        with gr.Row() as row:
            with gr.Column() as col:
                correct_drg_text_2 = gr.Textbox(visible=False, label="Input Correct DRG", interactive=True)
                correct_drg_text_2.submit(save_results, inputs=correct_drg_text_2, outputs=correct_drg_text_2)
                correct_drg_text_list.append(correct_drg_text_2)            
            with gr.Column() as col:
                salient_words_box_2 = gr.Textbox(visible=False, label="Input Salient Words (comma separated)", interactive=True)
                salient_words_box_2.submit(save_results, inputs=salient_words_box_2, outputs=salient_words_box_2)
                correct_salient_words_list.append(salient_words_box_2)
            attn_clr_2 = gr.ClearButton(value='Remove DRG Results', visible=False, components=[attn_2])    
            related_clr_bts.append(attn_clr_2)
            attn_clr_2.click(clr_btn, outputs=[attn_clr_2, correct_drg_text_2, salient_words_box_2]) 

        print("row 3")
        # related row 3
        with gr.Row() as row:
            attn_3 = gr.HTML()
            related_attn.append(attn_3)
        with gr.Row() as row:
            with gr.Column() as col:
                correct_drg_text_3 = gr.Textbox(visible=False, label="Input Correct DRG", interactive=True)
                correct_drg_text_3.submit(save_results, inputs=correct_drg_text_3, outputs=correct_drg_text_3)
                correct_drg_text_list.append(correct_drg_text_3)            
            with gr.Column() as col:
                salient_words_box_3 = gr.Textbox(visible=False, label="Input Salient Words (comma separated)", interactive=True)
                salient_words_box_3.submit(save_results, inputs=salient_words_box_3, outputs=salient_words_box_3)
                correct_salient_words_list.append(salient_words_box_3)
            attn_clr_3 = gr.ClearButton(value='Remove DRG Results', visible=False, components=[attn_3])    
            related_clr_bts.append(attn_clr_3)
            attn_clr_3.click(clr_btn, outputs=[attn_clr_3, correct_drg_text_3, salient_words_box_3]) 

        print("row 4")
        # input to related summaries
        with gr.Row() as row:
            input_related = gr.TextArea(label="Input up to 3 Related Discharge Summaries Here", visible=False, text_align='left', min_width=300)
        with gr.Row() as row:
            rmv_related_btn = gr.Button(value='Remove Related Summary', visible=False)
            # sbm_btn = gr.Button(value="Submit Related Summaries", components=[input_related], visible=False)         

                     
        print("row 5")
        with gr.Row() as row:
            related = gr.DataFrame(
                value=None, headers=['Similarity Score', 'Related Discharge Summary'], row_count=5,
                datatype=['number', 'str'], col_count=(2, 'fixed'), visible=False
            )
        # initial run
        btn.click(run, inputs=[input], outputs=[
            attn_viz, related, attn_clr_btn, input_related,
            rmv_related_btn, correct_drg_text, salient_words_box
        ])
        # find related summaries
        # related.click(load_example, inputs=[related], outputs=[input_related])
        related.select(load_df_example, inputs=[related], outputs=[input_related])
        # remove related summaries
        rmv_related_btn.click(remove_most_recent, outputs=[input_related])

        # # perform attribution on related summaries
        # sbm_btn.click(run_related, outputs=related_attn + related_clr_bts + correct_drg_text_list + correct_salient_words_list)

        
    demo.launch()

if __name__ == "__main__":
    main()