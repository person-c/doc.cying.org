import subprocess
import os
path = ['web', 'note', 'slides']

for p in path:
    f = open(f'{p}.rmd', 'x')
    f.write(f'''---
css: [static/css/link.css, static/css/font.css, static/css/list.css]
---

<header>
      <div class="nav-bg"></div>
      <nav class="nav-bar">
        <ul>
          <li><a href="index.html">Home</a></li>
          <li><a href="slides.html">Slides</a></li>
          <li><a href="web.html">Web</a></li>
          <li><a href="note.html">Note</a></li>
        </ul>
      </nav>
</header>

:::{{.blog-list}}
```{{r results='asis'}}
#| echo: false
if ("{p}" == "slides") {{
    slides_dir <- list.dirs(path = "resource/slides", recursive = FALSE)
    sprintf("<a href='%s/index.html'>%s</a>", slides_dir,
    gsub("resource/slides/", "", slides_dir)) |>
    cat(sep = "\\n\\n")
}} else {{
    file <- list.files(path = "resource/{p}",
        pattern = "*.html", recursive = TRUE)
    sprintf("<a href='resource/{p}/%s'>%s</a>",
      file, gsub(".html", "", file)) |>
    cat(sep = "\\n\\n")
}}
```
:::'''
            )
    f.close()

    subprocess.run(f'Rscript -e \'knitr::knit("{p}.rmd")\'', shell=True)
    subprocess.run(['pandoc', f'{p}.md',
                    '-o', f'{p}.html',
                    '--template=templates/list.html'])
    os.remove(f'{p}.rmd')
    os.remove(f'{p}.md')
