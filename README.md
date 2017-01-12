# Pix2Pix-Film
An implementation of [Pix2Pix](https://arxiv.org/abs/1611.07004) in Tensorflow for use with colorizing and increasing the field of view in frames from classic films. For more information, see my [Medium Post](https://medium.com/p/f4d551fa0503) on the project.

Pretrained model available [here](https://drive.google.com/open?id=0B8x0IeJAaBccNFVQMkQ0QW15TjQ). It was trained using Alfred Hitchcock films, so it generalizes best to similar movies.

To generate frames, use `ffmpeg` to resize and create `.png` frames from video.

* To resize video: `ffmpeg -strict -2 -i input.mp4 -vf scale=256:144 output.mp4 -strict -2`

* To generate frames: ` ffmpeg -i output.mp4 -vf fps=10 ./frames/out%6d.png`

In order to 'remaster' frames, run Pix2Pix iPython notebook.

To convert 'remastered' frames  back into a video, use:
`ffmpeg -framerate 10 -i frame%01d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4`
