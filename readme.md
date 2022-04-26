# Python OpenCV auto recognize wall data from battlemap image for Foundry dynamic light
image for recognize need place toghether with main file

image proccesing: resize, grayscale, gaussian blur, canny edge detection, find_countors,approximate polygone, delete similar lines, export to json scene data

![preview](https://media.discordapp.net/attachments/338735742236753930/968399360746455060/unknown.png?width=617&height=468)

because opencv dosen't support buttons in python for save file your need use slider (just pull it left or right)

for import wall map your need

1. Generate Data.json file
2. Open your foundry and in scene section create empty scene
3. RBM to your scene and select  ![Import Data](https://media.discordapp.net/attachments/338735742236753930/968402220846551050/unknown.png)
4. Set image(the same one you used) for scene background
5. Set padding to 0%
6. fix wall mistakes if your need it
7. profit)

Final result
![final](https://media.discordapp.net/attachments/338735742236753930/968403335189250068/unknown.png?width=805&height=468)