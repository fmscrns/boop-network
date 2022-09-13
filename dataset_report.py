import os, pathlib, glob, csv, xlsxwriter
from tqdm import tqdm


def glob_pattern(str):
    pattern = ""
    for x in str:
        if x == " ":
            pattern += "*"
        else:
            pattern += "[{}{}]".format(x.lower(), x.upper())
    return pattern

with tqdm(total=100) as pbar:
    datasets = glob.glob(os.getcwd() + "/external/*/", recursive=True)

    workbook = xlsxwriter.Workbook("dataset_report.xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.merge_range(0, 0, 3, 0, "Dog Breed Names")
    worksheet.merge_range(0, 1, 0, len(datasets)*4, "Dog Image Datasets")
    worksheet.merge_range(0, (len(datasets)*4)+1, 3, (len(datasets)*4)+1, "Total Image Count")
    worksheet.merge_range(0, (len(datasets)*4)+2, 3, (len(datasets)*4)+2, "Annotated Ratio")

    pbar.update(10)
    pbar.set_description("Creating workbook...")

    for c, ds in enumerate(datasets):
        ds_name = pathlib.PurePath(ds).name
        worksheet.merge_range(1, (c*4)+1, 1, (c*4)+4, ds_name)
        worksheet.merge_range(2, (c*4)+1, 2, (c*4)+3, "Image Count")
        worksheet.write(3, (c*4)+1, "Annotated")
        worksheet.write(3, (c*4)+2, "Unannotated")
        worksheet.write(3, (c*4)+3, "Total")
        worksheet.merge_range(2, (c*4)+4, 3, (c*4)+4, "ID")

        pbar.update(10/len(datasets))
        pbar.set_description("Importing dataset '{}'...".format(ds_name))

    with open('external/dog_breeds.csv') as csv_file, pbar as pbar:
        csv_len = sum(1 for line in csv_file)
        csv_file.seek(0)
        csv_reader = csv.reader(csv_file, delimiter=',')
        for c1, bn in enumerate(csv_reader, start=4):
            worksheet.write(c1, 0, bn[0])

            total_img_count = 0
            for c2, ds in enumerate(datasets):
                ds_path = pathlib.PurePath(ds)
                ds_name = ds_path.name
                # worksheet.merge_range(1, (c*4)+1, 1, (c*4)+4, ds_name)

                files = []
                img_ext = ["png", "jpg", "jpeg"]

                [
                    files.extend(
                        glob.glob(
                            ds + "/**/*{}*/*.{}".format(
                                glob_pattern(bn[0]), 
                                e
                            ),
                            recursive = True
                        )
                    ) for e in img_ext
                ]
                
                total_img_count += len(files)
                worksheet.write(c1, (c2*4)+3, len(files))

                pbar.update(80/(csv_len*len(datasets)))
                pbar.set_description("Writing data for breed '{}' and dataset '{}'...".format(bn[0], ds_name))

            worksheet.write(c1, (len(datasets)*4)+1, total_img_count)

    workbook.close()


# cd venv/lib/python3.7/site-packages/labelImg/