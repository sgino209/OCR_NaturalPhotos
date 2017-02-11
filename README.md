# OCR_NaturalPhotos
OCR of Natural Photos using Machine-Learning techniques (based on Python sklearn toolkit) 

Run Command:
python main.py -t [train_img_base] -x [test_img_base] -p [pickle_path] -c [cm_dump_path]

Optional flags: --cv_engine, --training_en, --no_classification, --no_pickle, --plot_en, --knn_study_en

Note:
On MacOS you might need to set LC_ALL and LANG for UTF-8:
env LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 python main.py -t [train_img_base] -x [test_img_base] -p [pickle_path] -c [cm_dump_path]

Charls74 Database was used for developing this environment:
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
