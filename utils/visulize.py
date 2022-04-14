import matplotlib.pyplot as plt
import io
from PIL import Image
import math


spanrow_image_template = '\t<td rowspan=\"{}\" width=\"220\" align=\"center\" valign=\"middle\">\
    <table>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
        <tr><td><img height=\"220\" width=\"220\" src=\"{}\"></td></tr>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
    </table></td>\n'

spanrow_video_template = '\t<td rowspan=\"{}\" width=\"220\" align=\"center\" valign=\"middle\">\
    <table>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
        <tr><td><video height=\"220\" width=\"220\" controls="" name="media"><source src="{}" type="video/mp4"></video></td></tr>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
    </table></td>\n'

spanrow_text_template = '\t<td rowspan=\"{}\" width=\"220\" align=\"center\" valign=\"middle\">{}</td>\n'

image_template = '\t<td width=\"220\" align=\"center\" valign=\"middle\">\
    <table>\
		<tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
		<tr><td><img height=\"220\" width=\"220\" src=\"{}\"></td></tr>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
	</table></td>\n'

red_image_template = '\t<td width=\"220\" align=\"center\" valign=\"middle\">\
    <table bordercolor="red" border="1" cellspacing="0">\
		<tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
		<tr><td><img height=\"220\" width=\"220\" src=\"{}\"></td></tr>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
	</table></td>\n'

video_template = '\t<td width=\"220\" align=\"center\" valign=\"middle\">\
    <table>\
		<tr><td height=\"40\" width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
		<tr><td><video height=\"220\" width=\"220\" controls="" name="media"><source src="{}" type="video/mp4"></video></td></tr>\
        <tr><td height=\"140\" width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
	</table></td>\n'

red_video_template = '\t<td width=\"220\" align=\"center\" valign=\"middle\">\
    <table bordercolor="red" border="1" cellspacing="0">\
		<tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
		<tr><td><video height=\"220\" width=\"220\" controls="" name="media"><source src="{}" type="video/mp4"></video></td></tr>\
        <tr><td width=\"220\" align=\"center\" valign=\"middle\" style=\"word-break:break-all\">{}</td></tr>\
	</table></td>\n'


empty_content = '<tr>\n\t<td style=\"border-left-style:hidden;border-right-style:hidden;\" \
    height=\"40\" width=\"220\" align=\"center\" valign=\"middle\"></td>\n</tr>\n'


def draw_block(body, head, w):
    num_line = int(len(body))
    content = ''

    head_flag = True if head is not None else False
    if head is not None:
        if head['type'] == 'video':
            head_template = spanrow_video_template
        elif head['type'] == 'text':
            head_template = spanrow_text_template
        elif head['type'] == 'image':
            head_template = spanrow_image_template
    
    for line in body:
        content += "<tr>\n"

        for item in line:
            if item['red']:
                body_template = red_video_template if item['type'] == 'video' else red_image_template
            else:
                body_template = video_template if item['type'] == 'video' else image_template

            if head_flag:
                if head['type'] == 'video' or head['type'] == 'image':
                    content += head_template.format(
                        num_line, head['title'], head['url'], head['text'])
                elif head['type'] == 'text':
                    content += head_template.format(
                        num_line, head['text'])
                head_flag = False
            
            content += body_template.format(item['title'], item['url'], item['text'])
        
        content += "</tr>\n"
    
    content += empty_content
    
    w.write(content)


def draw_page(save, info, size_per_line=-1):
    with open(save, 'w', encoding='utf-8') as w:
        w.write('<html>\n')
        w.write('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n')
        w.write('<body>\n')
        w.write('<table border="1" cellspacing="0" cellpadding="0">\n')
        
        for item in info:
            head = item['head']
            body = [item['body']] if size_per_line == -1 else [
                item['body'][i*size_per_line:(i+1)*size_per_line] for i in range(
                    math.ceil(len(item['body'])*1.0/size_per_line)
                )]
            draw_block(body, head, w)

        w.write('</table>\n')
        w.write('</body>\n')
        w.write('</html>\n')


def draw_video_to_videos(qid, qurl, qtext, gids, scores, gurls, gtexts, gspecials):
    draw_data = {}
    
    head_data = {
        'title': qid,
        'url': qurl,
        'text': qtext,
        'type': 'video'
    }

    body_data = [{'title': '{} {:.4f}'.format(gid, s),
                  'url': gurl, 
                  'text': gtext,
                  'red': gspecial,
                  'type':'video'} for gid, s, gurl, gtext, gspecial in zip(gids, scores, gurls, gtexts, gspecials)]

    draw_data['head'] = head_data
    draw_data['body'] = body_data

    return draw_data


def draw_text_to_videos(qtext, gids, gurls, gtexts, gauthors, gscores, gspecials):
    draw_data = {}
    
    head_data = {
        'text': qtext,
        'type': 'text'
    }

    body_data = [{'title': '{}</br>{}</br>{:.5f}'.format(gid, gauthor, gscore),
                  'url': gurl, 
                  'text': gtext,
                  'red': gspecial,
                  'type':'video'} for gid, gurl, gtext, gauthor, gscore, gspecial in zip(gids, gurls, gtexts, gauthors, gscores, gspecials)]

    draw_data['head'] = head_data
    draw_data['body'] = body_data

    return draw_data


def draw_videos_tag(vids, urls, texts, tags, scores):
    draw_data = {}

    body_data = []
    for vid, url, text, tag, score in zip(vids, urls, texts, tags, scores):
        vistext = []
        for t, s in zip(tag, score):
            vistext.append('{}: {:.6f}'.format(t, s))
        body_data.append(
            {'title': vid,
             'url': url, 
             'text': '</br>'.join(vistext),
             'red': False,
             'type':'video'}
        )

    draw_data['head'] = None
    draw_data['body'] = body_data

    return draw_data


def draw_image_to_videotags(qtag, qtagimage, vids, urls, texts, tags, scores):
    draw_data = {}

    head_data = {
        'title': qtag,
        'url': qtagimage,
        'text': '',
        'type': 'image'
    }

    body_data = []
    for vid, url, text, tag, score in zip(vids, urls, texts, tags, scores):
        vistext = []
        for t, s in zip(tag, score):
            vistext.append('{}: {:.6f}'.format(t, s))
        body_data.append(
            {'title': vid,
             'url': url, 
             'text': '</br>'.join(vistext),
             'red': False,
             'type':'video'}
        )

    draw_data['head'] = head_data
    draw_data['body'] = body_data

    return draw_data


def draw_sim_mat(sim_mat):
    max_size = sim_mat.shape[0]

    plt.figure(figsize=(max_size, max_size))
    plt.imshow(sim_mat, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    for i in range(max_size):
        for j in range(max_size):
            plt.text(j, i, '{:.3f}'.format(sim_mat[i, j]), ha='center', va='center', color='w')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return image
