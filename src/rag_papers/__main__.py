# In Python, __main__.py is a special file.
# If a folder is a package (it has __init__.py), and it also has a __main__.py, then Python allows:
# python -m <package_name>
# So now:
# package name: rag_papers
# entry file: rag_papers/__main__.py

# This file allows running the package as:
# python -m rag_papers
# “When someone runs the package, just call the CLI’s main function.”

from .cli import main

if __name__ == "__main__":
    main()
