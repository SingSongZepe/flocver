# Flocver

#### python dependencies
- opencv-python
- numpy
- pyside6 (optional, just use for showing picture in a widget, you can also use show_one_image() function to do that)

#### Run
cmd\> python ./main.py
> if you use ***vscode*** to run the program, then just press ctrl + F5.

#### Alternative
call different function to get different result image

Algorithm:
> module **blur**:
> > cvt_gaussain_blur
> > cvt_mean_blur
> > cvt_bilateral_blur

> module **morphology**:
> > morphology

> module **convert2gray**
> > convert2gray
> > convert2

> module **cvt_color**
> > cvt_color

#### Change Background cvt_color
In main(), there be flocver.cvt_color() function,
change it's input, you can use any color(in three dimension, e.g. (135,206,235))