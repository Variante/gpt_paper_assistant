import json
import random
from datetime import datetime
import re

link_prefix = 'user-content-'
topic_shift = 1000

def render_paper(paper_entry: dict, idx: int) -> str:
    """
    :param paper_entry: is a dict from a json. an example is
    {"paperId": "2754e70eaa0c2d40972c47c4c23210f0cece8bfc", "externalIds": {"ArXiv": "2310.16834", "CorpusId": 264451832}, "title": "Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution", "abstract": "Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel discrete score matching loss that is more stable than existing methods, forms an ELBO for maximum likelihood training, and can be efficiently optimized with a denoising variant. We scale our Score Entropy Discrete Diffusion models (SEDD) to the experimental setting of GPT-2, achieving highly competitive likelihoods while also introducing distinct algorithmic advantages. In particular, when comparing similarly sized SEDD and GPT-2 models, SEDD attains comparable perplexities (normally within $+10\\%$ of and sometimes outperforming the baseline). Furthermore, SEDD models learn a more faithful sequence distribution (around $4\\times$ better compared to GPT-2 models with ancestral sampling as measured by large models), can trade off compute for generation quality (needing only $16\\times$ fewer network evaluations to match GPT-2), and enables arbitrary infilling beyond the standard left to right prompting.", "year": 2023, "authors": [{"authorId": "2261494043", "name": "Aaron Lou"}, {"authorId": "83262128", "name": "Chenlin Meng"}, {"authorId": "2490652", "name": "Stefano Ermon"}], "ARXIVID": "2310.16834", "COMMENT": "The paper shows a significant advance in the performance of diffusion language models, directly meeting one of the criteria.", "RELEVANCE": 10, "NOVELTY": 8}, "2310.16779": {"paperId": "edc8953d559560d3237fc0b27175cdb1114c0ca5", "externalIds": {"ArXiv": "2310.16779", "CorpusId": 264451949}, "title": "Multi-scale Diffusion Denoised Smoothing", "abstract": "Along with recent diffusion models, randomized smoothing has become one of a few tangible approaches that offers adversarial robustness to models at scale, e.g., those of large pre-trained models. Specifically, one can perform randomized smoothing on any classifier via a simple\"denoise-and-classify\"pipeline, so-called denoised smoothing, given that an accurate denoiser is available - such as diffusion model. In this paper, we investigate the trade-off between accuracy and certified robustness of denoised smoothing: for example, we question on which representation of diffusion model would maximize the certified robustness of denoised smoothing. We consider a new objective that aims collective robustness of smoothed classifiers across multiple noise levels at a shared diffusion model, which also suggests a new way to compensate the cost of accuracy in randomized smoothing for its certified robustness. This objective motivates us to fine-tune diffusion model (a) to perform consistent denoising whenever the original image is recoverable, but (b) to generate rather diverse outputs otherwise. Our experiments show that this fine-tuning scheme of diffusion models combined with the multi-scale smoothing enables a strong certified robustness possible at highest noise level while maintaining the accuracy closer to non-smoothed classifiers.", "year": 2023, "authors": [{"authorId": "83125078", "name": "Jongheon Jeong"}, {"authorId": "2261688831", "name": "Jinwoo Shin"}], "ARXIVID": "2310.16779", "COMMENT": "The paper presents an advancement in the performance of diffusion models, specifically in the context of denoised smoothing.", "RELEVANCE": 9, "NOVELTY": 7}
    :return: a markdown formatted string showing the arxiv id, title, arxiv url, abstract, authors, score and comment (if those fields exist)
    """
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"]
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # get the abstract
    abstract = paper_entry["abstract"]
    # get the authors
    authors = paper_entry["authors"]
    paper_string = f'### {idx}\. [{title}]({arxiv_url})\n'
    paper_string += f"**ArXiv:** {arxiv_id} [[page]({arxiv_url})] [[pdf]({arxiv_pdf_url})]\n\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n\n'
    paper_string += f"**Abstract:** {abstract}\n\n"
    if "COMMENT" in paper_entry:
        comment = paper_entry["COMMENT"]
        paper_string += f"**Comment:** {comment}\n\n"
    if "RELEVANCE" in paper_entry:
        paper_string += f"**Relevance:** {paper_entry['RELEVANCE']}\n"
    topic_id = idx // topic_shift 
    topic_str = f'topic-{topic_id}' if topic_id else 'go-beyond'
    paper_string += f"Back to [[topic](#{link_prefix}{topic_str})] [[top](#{link_prefix}topics)]\n"
    return paper_string


def render_title_and_author(paper_entry: dict, idx: int) -> str:
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"]
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    authors = paper_entry["authors"]
    
    raw_title_url = f'{idx} {title}'
    # Keep only English letters, numbers, and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9 -]', '', raw_title_url)
    
    # Replace spaces with dashes
    cleaned = cleaned.replace(' ', '-').lower()
    paper_string = f'{idx}\. [{title}]({arxiv_url}) [[more](#{link_prefix}{cleaned})] \\\n'
    paper_string += f'**Authors:** {", ".join(authors)}\n'
    return paper_string


def render_criteria(criteria: list[str]) -> str:
    criteria_string = ""
    for criterion in criteria:
        topic_idx = int(criterion.split('.')[0])
        criteria_string += f"[{criterion}](#{link_prefix}topic-{topic_idx})\n\n"
    criteria_string += f'[Go beyond](#{link_prefix}go-beyond)\n\n'
    return criteria_string

def extract_criterion_from_paper(paper_entry: dict) -> int:
    if "COMMENT" not in paper_entry:
        return 0
    # Regular expression pattern to find 'criterion' followed by a number
    pattern = r'riteri(.+?) (\d+)'
    # Search for the pattern in the text
    match = re.search(pattern, paper_entry["COMMENT"])
    if match:
        # Extract the number (group 1 in the match)
        criterion_number = match.group(2)
        return int(criterion_number)
    else:
        return 0 # not sure

def render_md_paper_title_by_topic(topic, paper_in_topic: list[str]) -> str: 
    return f"### {topic}\n" +  "\n".join(paper_in_topic) + f"\n\nBack to [[top](#{link_prefix}topics)]\n\n---\n"
        

def render_md_string(papers_dict):
    # header
    with open("configs/paper_topics.txt", "r") as f:
        criteria = f.readlines()
        
    filtered_criteria = [i for i in criteria if len(i.strip()) and i.strip()[0] in '0123456789']
    
    criteria_string = render_criteria(filtered_criteria)
        
    def generate_background_for_white_foreground(threshold:int=150):
        # Ensure that the color is dark enough for white text to be readable
        # by keeping the RGB values below a certain threshold (e.g., 200)
        r = random.randint(0, threshold)
        g = random.randint(0, threshold)
        b = random.randint(0, threshold)

        # Convert the RGB values to a hexadecimal string
        hex_color = f'{r:02x}{g:02x}{b:02x}'
        return hex_color
        
    random_font = random.sample(['Cookie', 'Lato', 'Arial', 'Comic', 'Inter', 'Bree', 'Poppins'], 1)[0]
    random_emoji = random.sample("😸😺🐱🐶🐼🐰🐥🐢🐣🌸🍀🌈☀️🍓🍦🍪🐻🦊🦄🐣🐤🐦🐧🦉🐸🐞🦋🐝🍄🌻🌷🌼🌺🌿🍃🍒🍑🍍🍉🍌🍫🍬🍭🍯🍼🧸🎀🎈🎉🛀🎁💐💖💕💞💓💗💘💝💟💌🎊🍩🍨🍧🍡🍖🍗🍕🍔🍟🌭🍿🍱🍣🍤🍥🍚🍙🍘🍜🍝🍛🍢🍵🍶🥂🥤🍹🍺🍻🥃🍷🥄🍴🍽🥢🥡🥪🥗🥘🥓🥞🥐🥖🥨🥯🥚🥦🥒🥑🥔🥕🥗🥝🥭🥥🥬🥪🥫🥟🥠🥡🥢🥣🥤🥧🥨🥩🥪🥫🥬🥭🥮", 1)[0]
     
    output_string = (
        "# Personalized Daily Arxiv Papers "
        + datetime.today().strftime("%m/%d/%Y")
        + "\n\nThis project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)\n\n"
        + "About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:\n\n"
        + f"""<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT-4 cost&emoji={random_emoji}&slug=Variante&button_colour={generate_background_for_white_foreground()}&font_colour=ffffff&font_family={random_font}&outline_colour=000000&coffee_colour=FFDD00" /></a>\n"""
        + "\n\n## Topics\n\nPaper selection prompt and criteria (jump to the section by clicking the link):\n\n"
        + criteria_string
        + "\n---\n"
        # + "## All\n Total relevant papers: "
        # + str(len(papers_dict))
        # + "\n\n"
        # + "Table of contents with paper titles:\n\n"
    )
    '''
    title_strings = [
        render_title_and_author(paper, i)
        for i, paper in enumerate(papers_dict.values())
    ]
    # output_string = output_string + "\n".join(title_strings) + "\n---\n"
    '''
    # render each topic
    paper_title_group_by_topic = [[] for _ in range(len(filtered_criteria) + 1)]
    paper_full_group_by_topic = [[] for _ in range(len(filtered_criteria) + 1)]
    for i, paper in enumerate(papers_dict.values()):
        paper_topic_idx = extract_criterion_from_paper(paper)
        if paper_topic_idx > len(filtered_criteria):
            paper_topic_idx = 0
        title_string = render_title_and_author(paper, i + paper_topic_idx * topic_shift)
        paper_title_group_by_topic[paper_topic_idx].append(title_string)
        full_string = render_paper(paper, i + paper_topic_idx * topic_shift)
        paper_full_group_by_topic[paper_topic_idx].append(full_string)
        
    for topic_idx, paper_in_topic in enumerate(paper_title_group_by_topic):
        if topic_idx == 0:
            # unknown topic
            continue
        output_string += render_md_paper_title_by_topic(f'Topic {topic_idx}', paper_in_topic) 
    output_string += render_md_paper_title_by_topic("Go beyond", paper_title_group_by_topic[0])

    """
    # render each paper
    paper_strings = [
        render_paper(paper, i) for i, paper in enumerate(papers_dict.values())
    ]
    """
    paper_string = "\n---\n".join(["\n".join(paper_in_topic) for paper_in_topic in paper_full_group_by_topic[1:] + paper_full_group_by_topic[:1] if len(paper_in_topic)])
    # join all papers into one string
    output_string += f"## Full paper list\n {paper_string}"
    # output_string += "\n\n---\n\n"
    # output_string += f"## Paper selection prompt\n{criterion}"
    return output_string


if __name__ == "__main__":
    # parse output.json into a dict
    with open("out/output.json", "r") as f:
        output = json.load(f)
    # write to output.md
    with open("out/output.md", "w") as f:
        f.write(render_md_string(output))
