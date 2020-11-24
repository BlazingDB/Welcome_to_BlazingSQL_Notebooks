from IPython.display import display, HTML
from pygments import highlight
from pygments.lexers.python import Python3Lexer
from pygments.formatters import HtmlFormatter

def print_code(code):
    template = """<style>
    {}
    </style>
    {}
    """

    lexer = Python3Lexer()
    formatter = HtmlFormatter(cssclass='pygments')

    html_code = highlight(code, lexer, formatter)
    css = formatter.get_style_defs('.pygments')

    html = template.format(css, html_code)
    return display(HTML(html))