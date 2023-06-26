from sentence_transformers import SentenceTransformer
import scipy
from tqdm import tqdm

def get_1_1_alignments(
    complex_file, 
    simple_file,
    output_file,
    model_name='T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
    min_similarity = 0.9,
    progress_bar = False
):
    def get_alignments(com_doc_lines, sim_doc_lines, transformerEmbedder):
        complexEmbedding = transformerEmbedder.encode(com_doc_lines, show_progress_bar=False)
        simpleEmbeddings  = transformerEmbedder.encode(sim_doc_lines, show_progress_bar=False)
        alignments_ = []

        for query, query_embedding in zip(com_doc_lines, complexEmbedding):
            distances = scipy.spatial.distance.cdist([query_embedding], simpleEmbeddings, "cosine")[0]
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            for idx, distance in results[0:1]:
                if(1-distance > min_similarity):
                    alignments_.append((query, sim_doc_lines[idx], 1-distance))

        return alignments_
    
    transformerEmbedder = SentenceTransformer(model_name)
    
    with open(complex_file, 'r') as comf:
        with open(simple_file, 'r') as simf:
            com_lines = comf.readlines()
            sim_lines = simf.readlines()

            com_doc_lines = []
            com_all_docs_lines = []
            for com_line in com_lines:
                if '.eoa' in com_line:
                    com_all_docs_lines.append(com_doc_lines)
                    com_doc_lines = []
                else:
                    com_doc_lines.append(com_line.strip())

                    
            sim_doc_lines = []
            sim_all_docs_lines = []
            for sim_line in sim_lines:
                if '.eoa' in sim_line:
                    sim_all_docs_lines.append(sim_doc_lines)
                    sim_doc_lines = []
                else:
                    sim_doc_lines.append(sim_line.strip())
                    

            assert len(com_all_docs_lines) == len(sim_all_docs_lines), "Number of docs in complex file is not equal to the number of docs in simple file"
    
            
    alignments_all = []
    
    with tqdm(total=len(com_all_docs_lines)) as pbar:
        for com_doc_lines, sim_doc_lines in zip(com_all_docs_lines, sim_all_docs_lines):
            alignments = get_alignments(com_doc_lines, sim_doc_lines, transformerEmbedder)
            alignments_all.extend(alignments)
            if progress_bar:
                pbar.update()
        
    with open(output_file+'.complex', 'w') as outcf:
        with open(output_file+'.simpl', 'w') as outsf:
            for entry in alignments_all:
                outcf.write(entry[0]+'\n')
                outsf.write(entry[1]+'\n')
        
    return alignments_all, output_file+'.complex', output_file+'.simpl'