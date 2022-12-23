from PIL import Image

import numpy as np
import wordcloud
import settings

keywords = ['下界','世间','东土', '仙女','众僧','众神','兄弟','八戒','公主','公主道','兵器','凤仙郡','凶神','化斋',
 '和尚', '唐僧', '国王', '圈子', '土地', '圣旨', '外公', '天兵', '天尊', '天神', '太子', '女儿', '女婿', '女子',
 '妖王', '妖道', '妖邪', '妖魔', '娘娘', '官道', '宝殿', '宝贝', '山神', '巡山', '巨灵', '师父', '徒弟', '性命',
 '怪物', '手段', '护国', '朱紫国', '李天王', '模样', '毫毛', '水面', '沙僧', '浴池', '父子', '狮子', '狮驼', '猴儿',     
 '玉兔', '玉帝', '王母', '皇后', '皇帝', '祸事', '罗汉', '群猴', '老儿', '老君', '老妖', '老爷', '老者', '老道',
 '老高', '老魔', '肚里', '苍蝇', '菩萨', '葫芦', '行李', '袈裟', '观音菩萨', '谢恩', '贤弟', '败阵', '那郡侯' '金箍棒',
 '钵盂', '钻风', '铁棒', '铁笼', '铃儿', '长官', '长老', '马匹', '马来', '龙王']

txt = ' '.join(keywords)
background = np.array(Image.open('./src/孙悟空.png'))
w = wordcloud.WordCloud(background_color="white",
                    font_path=settings.FONT_PATH, 
                    width=240, height=600,
                    max_words=2000, mask=background)
w.generate(txt)
w.to_file(f"pywcloud.png")