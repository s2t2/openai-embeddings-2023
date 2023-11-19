

import re




def convert_non_ascii(txt):
    # we see tokens like:
    #   'état', 'être',
    # 'últimahora', 'μολωνλαβε', 'قاسم_سليماني', '𝐔𝐍𝐇𝐈𝐍𝐆𝐄𝐃', '𝐜𝐨𝐮𝐧𝐭𝐫𝐲',
    # '𝐝𝐚𝐲𝐬', '𝐨𝐮𝐫', '𝐩𝐨𝐥𝐢𝐭𝐢𝐜𝐬', '𝐮𝐬', '𝑤𝑒𝑎𝑘𝑒𝑠𝑡', '𝑱𝑶𝑯𝑵', '𝑹𝑶𝑩𝑬𝑹𝑻𝑺',
    # '𝔽𝕆ℝ𝔼𝕍𝔼ℝ', '𝕋𝕙𝕒𝕥', '𝕖𝕞𝕓𝕒𝕣𝕣𝕒𝕤𝕤𝕚𝕟𝕘', '𝕛𝕦𝕤𝕥', '𝗖𝗿𝗼𝗽𝘀', '𝗗𝗡𝗖', '𝗗𝗮𝗶𝗹𝘆',
    # '𝗘𝗡𝗗𝗢𝗥𝗦𝗘𝗗', '𝗘𝗡𝗙𝗢𝗥𝗖𝗘𝗦', '𝗝𝗢𝗬', '𝗝𝗢𝗬𝗦', '𝗟𝗲𝗮𝗱𝗶𝗻𝗴', '𝗡𝗢', '𝗢𝗙',
    # '𝗣𝗹𝗮𝘆𝗯𝗼𝗼𝗸', '𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿', '𝗦𝘁𝗮𝘁𝗲𝘀', '𝗩𝗲𝗴𝗲𝘁𝗮𝗯𝗹𝗲𝘀', '𝗰𝗿𝗲𝗱𝗶𝗯𝗶𝗹𝗶𝘁𝘆',
    # '𝗳𝗼𝗿𝗲𝘃𝗲𝗿', '𝗶𝗺𝗽𝗲𝗮𝗰𝗵𝗲𝗱', '𝗶𝗻', '𝗶𝗻𝗲𝘃𝗶𝘁𝗮𝗯𝗹𝗲', '𝗻𝗲𝘃𝗲𝗿', '𝗻𝗼',
    # '𝙀𝙢𝙗𝙤𝙡𝙙𝙚𝙣', '𝙛𝙖𝙢𝙞𝙡𝙮', '𝙛𝙚𝙚𝙡', '𝙜𝙧𝙤𝙪𝙥', '𝙝𝙞𝙨', '𝙞𝙣', '𝙠𝙞𝙙𝙨', '𝙨𝙖𝙙'

    # so we'll convert to keep their meaning:

    terms_map = {
        'état': 'etat',
        'être': 'etre',
        'últimahora': 'ultimahora',
        'μολωνλαβε': 'molonlabe',
        'قاسم_سليماني': 'Qasem_Soleimani',
        '𝐔𝐍𝐇𝐈𝐍𝐆𝐄𝐃': 'UNHINGED',
        '𝐜𝐨𝐮𝐧𝐭𝐫𝐲': 'country',
        '𝐝𝐚𝐲𝐬': 'days',
        '𝐨𝐮𝐫': 'our',
        '𝐩𝐨𝐥𝐢𝐭𝐢𝐜𝐬': 'politics',
        '𝐮𝐬': 'us',
        '𝑤𝑒𝑎𝑘𝑒𝑠𝑡': 'weakest',
        '𝑱𝑶𝑯𝑵': 'JOHN',
        '𝑹𝑶𝑩𝑬𝑹𝑻𝑺': 'ROBERTS',
        '𝔽𝕆ℝ𝔼𝕍𝔼ℝ': 'FOREVER',
        '𝕋𝕙𝕒𝕥': 'That',
        '𝕖𝕞𝕓𝕒𝕣𝕣𝕒𝕤𝕤𝕚𝕟𝕘': 'embarrassing',
        '𝕛𝕦𝕤𝕥': 'just',
        '𝗖𝗿𝗼𝗽𝘀': 'Crops',
        '𝗗𝗡𝗖': 'DNC',
        '𝗗𝗮𝗶𝗹𝘆': 'Daily',
        '𝗘𝗡𝗗𝗢𝗥𝗦𝗘𝗗': 'ENDORSED',
        '𝗘𝗡𝗙𝗢𝗥𝗖𝗘𝗦': 'ENFORCES',
        '𝗝𝗢𝗬': 'JOY',
        '𝗝𝗢𝗬𝗦': 'JOYS',
        '𝗟𝗲𝗮𝗱𝗶𝗻𝗴': 'Leading',
        '𝗡𝗢': 'NO',
        '𝗢𝗙': 'OF',
        '𝗣𝗹𝗮𝘆𝗯𝗼𝗼𝗸': 'Playbook',
        '𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿': 'Reminder',
        '𝗦𝘁𝗮𝘁𝗲𝘀': 'States',
        '𝗩𝗲𝗴𝗲𝘁𝗮𝗯𝗹𝗲𝘀': 'Vegetables',
        '𝗰𝗿𝗲𝗱𝗶𝗯𝗶𝗹𝗶𝘁𝘆': 'credibility',
        '𝗳𝗼𝗿𝗲𝘃𝗲𝗿': 'forever',
        '𝗶𝗺𝗽𝗲𝗮𝗰𝗵𝗲𝗱': 'impeached',
        '𝗶𝗻': 'in',
        '𝗶𝗻𝗲𝘃𝗶𝘁𝗮𝗯𝗹𝗲': 'inevitable',
        '𝗻𝗲𝘃𝗲𝗿': 'never',
        '𝙀𝙢𝙗𝙤𝙡𝙙𝙚𝙣': 'Embolden',
        '𝙛𝙖𝙢𝙞𝙡𝙮': 'family',
        '𝙛𝙚𝙚𝙡': 'feel',
        '𝙜𝙧𝙤𝙪𝙥': 'group',
        '𝙝𝙞𝙨': 'his',
        '𝙞𝙣': 'in',
        '𝙠𝙞𝙙𝙨': 'kids',
        '𝙨𝙖𝙙': 'sad',
        '𝗻𝗼': 'no',
        '𝙩𝙚𝙭𝙩': 'text',

        # these don't work ?:
        'zürich': 'zurich',
        'últimahora': 'ultimahora',
        'μολωνλαβε': 'molonlabe', # come and take them

    }

    for k, v in terms_map.items():
        txt = txt.replace(k, v)

    # despite best efforts, remove any remaining non-asci:

    non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
    txt = non_ascii_pattern.sub('', txt)

    return txt
