""" 
"Hello. world!" for DeepLens. 

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.
"""
from deeplens import GeoLens

def main():
    lens = GeoLens(filename='./datasets/lens_zoo/cellphone.json')
    lens.analysis(render=True)

if __name__=='__main__':
    main()