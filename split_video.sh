mkdir -p jacksontown && cd jacksontown
ffmpeg -i ../data/video0-JacksonTownSquareLiveWebcam.mp4 -c copy -map 0 -segment_time 3600 -reset_timestamps 1 -f segment "%01d.mp4"