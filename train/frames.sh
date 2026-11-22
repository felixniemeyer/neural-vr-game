#!/bin/sh
# Centre-crop square, scale to SIZE, FPS frames/s, drop first and last TRIM seconds.
# videos/*.mp4 -> data/frames/<SIZE>/<name>.mp4
SIZE=${SIZE:-128} FPS=${FPS:-10} TRIM=${TRIM:-300}
cd "$(dirname "$0")" && mkdir -p data/frames/$SIZE
for f in videos/*.mp4; do
  out=data/frames/$SIZE/$(basename "$f")
  [ -f "$out" ] && continue
  len=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  ffmpeg -v error -stats -ss $TRIM -i "$f" -t "$(echo "$len - 2*$TRIM" | bc)" -an -sn \
    -vf "crop=ih:ih,scale=$SIZE:$SIZE:flags=area,fps=$FPS" \
    -c:v libx264 -crf 12 -pix_fmt yuv444p "$out.part.mp4" && mv "$out.part.mp4" "$out"
done
