import os, shutil
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

def yolo_annot_conversion(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return "{} {} {} {}".format(x, y, w, h)

def yolo_annot(dog_breed_num, imgcont_path, oldcont_path, newcont_path):
    for dirpath, subdirs, files in os.walk(oldcont_path):
        for count, f in enumerate(tqdm(files)):
            # if count > 1:
            #     break

            shutil.copy2(
                os.path.join(oldcont_path, f),
                newcont_path
            )

            img_ext = [".jpg", ".jpeg"]
            remove_ext = [".xml"] + img_ext

            old_annot_path = os.path.join(newcont_path, f)
            img_path = os.path.join(imgcont_path, f)

            for ext in remove_ext:
                new_annot_path = old_annot_path.replace(ext, "")
                img_path = img_path.replace(ext, "")

            for ext in img_ext:
                if os.path.isfile(img_path + ext):
                    img_path += ext

            new_annot_path += ".txt"

            os.rename(
                old_annot_path,
                new_annot_path
            )

            with open(new_annot_path, 'r') as xmlfile:
                xml_content = xmlfile.read() 

            soup = BeautifulSoup(xml_content, "lxml")

            image = Image.open(img_path)
            image_width = int(image.size[0])
            image_height = int(image.size[1])

            bbox = soup.find_all("bodybndbox") + soup.find_all("bndbox")
            for b in bbox:
                xmin = int(b.find("xmin").text)
                ymin = int(b.find("ymin").text)
                xmax = int(b.find("xmax").text)
                ymax = int(b.find("ymax").text)

                yolo_annot = yolo_annot_conversion(
                    (image_width, image_height),
                    (xmin, xmax, ymin, ymax)
                )

                with open(new_annot_path, 'w') as xmlfile:
                    xmlfile.write(
                        "{} {}\n".format(
                            dog_breed_num,
                            yolo_annot
                        )
                    )

yolo_annot(
    dog_breed_num = 4,
    imgcont_path = "/Users/fmscrns/Desktop/boop-network/chosen_dogs/images/pug/stanford_dogs",
    oldcont_path = "/Users/fmscrns/Desktop/boop-network/chosen_dogs/annotations/pug/stanford_dogs",
    newcont_path = "/Users/fmscrns/Desktop/boop-network/database/pug (4)/stanford/annotations"
)