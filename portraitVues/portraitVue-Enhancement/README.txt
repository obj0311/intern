<Training>
path : 		training data path
test_path : 	valid data path
task : 		FaceInpainting --- mask까지 생성한 degradation 이미지
 		None --- mask 없이 degradation만 된 이미지
phase: 		'train'으로 설정

<Test>
path : 		training data path
test_path : 	test data path
task : 		'None'으로 설정하여 mask 없는 이미지 생성
phase: 		'test'으로 설정
pretrain: 		pretrained된 model의 path 설정

++ result_dir 저장할 위치 설정 ++

처음에 test dataset을 읽어오지 못했는데 이는 bmp 확장자로 되어있어서 못 읽었던것.
training/dataset_face.py class FaceDataset(Dataset): 선언 부분에서 def __init__의 self.HQ_imgs에 glob.glob(os.path.join(path, '*.bmp')) 추가하기