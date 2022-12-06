from PIL import Image

import numpy as np
import wordcloud
import settings

txt = "life is short,you need python"
background = np.array(Image.open('./src/template.png'))
w = wordcloud.WordCloud(background_color="white",
                    font_path=settings.FONT_PATH, 
                    width=240, height=600,
                    max_words=2000, mask=background)
w.generate(txt)
w.to_file(f"pywcloud.png")