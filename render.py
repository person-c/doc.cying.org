import os
import subprocess

file = os.listdir('.')
toc = [f for f in file if f.endswith('.Rmd')]

for f in toc:
    subprocess.run(f'Rscript -e "knitr::knit(\'{f}\')"', shell=True)
    subprocess.run(['pandoc', f, '-o', f.replace("Rmd", "html"),
                    '--template=templates/default.html'])
