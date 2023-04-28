import os
import re
import time
import copy
import codecs
import pickle
import numpy as np
import six
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import functional as F
from torchvision import transforms
import argparse
from mmcv import Config
from tqdm import tqdm
import lmdb

from cdistnet.data.transform import CVGeometry, CVColorJitter, CVDeterioration

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    # print('Load set vocabularies as %s.' % vocab)
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        for t in self.transform:
            image, target = t(image, target)
        return image, target


class TXTDataset(Dataset):

    def __init__(
            self,
            image_dir,
            gt_file,
            word2idx,
            idx2word,
            size=(100, 32),
            max_width=256,
            rgb2gray=True,
            keep_aspect_ratio=False,
            is_test=False,
            is_lower=False,
            data_aug=True,
    ):
        self.image_dir = image_dir
        self.gt = dict()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.is_test = is_test
        self.rgb2gray = rgb2gray
        self.width = size[0]
        self.height = size[1]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_width = max_width
        self.is_lower = is_lower
        self.data_aug = data_aug
        print("preparing data ...")
        print("path:{}".format(gt_file))
        with open(gt_file, 'r', encoding='UTF-8-sig') as f:
            all = f.readlines()
            for each in tqdm(all):
                each = each.strip().split(' ')
                image_name, text = each[0], ' '.join(each[1:])
                self.gt[image_name] = text
        self.data = list(self.gt.items())
        if self.is_test==False and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data=[(img_path,text),...]
        image_name = self.data[idx][0]
        image_path = os.path.join(self.image_dir, image_name)
        # print(image_path)
        if self.rgb2gray:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')
        h, w = image.height, image.width
        if self.is_test==False and self.data_aug:
            image = self.augment_tfs(image)
        if self.is_test:
            if h / w > 2:
                image1 = image.rotate(90, expand=True)
                image2 = image.rotate(-90, expand=True)
            else:
                image1 = copy.deepcopy(image)
                image2 = copy.deepcopy(image)
        if self.keep_aspect_ratio:
            h, w = image.height, image.width
            ratio = w / h
            image = image.resize(
                (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                Image.ANTIALIAS
            )
            if self.is_test:
                h, w = image1.height, image1.width
                ratio = w / h
                image1 = image1.resize(
                    (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                    Image.ANTIALIAS
                )
                image2 = image2.resize(
                    (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                    Image.ANTIALIAS
                )
        else:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
            if self.is_test:
                image1 = image1.resize((self.width, self.height), Image.ANTIALIAS)
                image2 = image2.resize((self.width, self.height), Image.ANTIALIAS)
        image = np.array(image)
        if self.is_test:
            image1 = np.array(image1)
            image2 = np.array(image2)
        if self.rgb2gray:
            image = np.expand_dims(image, -1)
            if self.is_test:
                image1 = np.expand_dims(image1, -1)
                image2 = np.expand_dims(image2, -1)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128. - 1.
        if self.is_test:
            image1 = image1.transpose((2, 0, 1))
            image2 = image2.transpose((2, 0, 1))
            image1 = image1.astype(np.float32) / 128. - 1.
            image2 = image2.astype(np.float32) / 128. - 1.
            image_final = np.concatenate([image, image1, image2], axis=0)

        text = self.data[idx][1]
        if self.is_lower:
            text = [self.word2idx.get(ch.lower(), 1) for ch in text]
        else:
            text = [self.word2idx.get(ch, 1) for ch in text]
        text.insert(0, 2)
        text.append(3)
        target = np.array(text)
        if self.is_test:
            return image_final, self.gt[image_name], image_name
        return image, target

class LMDBDataset(Dataset):

    def __init__(
            self,
            image_dir,
            gt_file,
            word2idx,
            idx2word,
            size=(100, 32),
            max_width=256,
            rgb2gray=True,
            keep_aspect_ratio=False,
            is_test=False,
            is_lower=False,
            data_aug=True,
    ):
        self.image_dir = image_dir
        self.gt = dict()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.is_test = is_test
        self.rgb2gray = rgb2gray
        self.width = size[0]
        self.height = size[1]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_width = max_width
        self.is_lower = is_lower
        self.data_aug = data_aug

        print("preparing data ...")
        print("path:{}".format(gt_file))
        self.env = lmdb.open(str(gt_file), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {gt_file}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))
        print("samples = {}".format(self.length))

        if self.is_test==False and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    def __len__(self):
        return self.length

    def get(self,idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx+1:09d}', f'label-{idx+1:09d}'
            label = str(txn.get(label_key.encode()), 'utf-8')  # label
            label = re.sub('가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됬됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항핳해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝힣ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890[^0-9a-zA-Z]+', '', label)
            label = label[:30]
            imgbuf = txn.get(image_key.encode())  # image
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            if self.rgb2gray:
                image = Image.open(buf).convert('L')
            else:
                image = Image.open(buf).convert('RGB')
            return image, label, image_key

    def __getitem__(self, idx):
        # self.data=[(img_path,text),...]
        image,label, idx = self.get(idx)

        if self.is_test==False and self.data_aug:
            image = self.augment_tfs(image)
        if self.keep_aspect_ratio:
            h, w = image.height, image.width
            ratio = w / h
            image = image.resize(
                (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                Image.ANTIALIAS
            )
        else:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        image = np.array(image)
        if self.rgb2gray:
            image = np.expand_dims(image, -1)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128. - 1.

        text = label
        if self.is_lower:
            text = [self.word2idx.get(ch.lower(), 1) for ch in text]
        else:
            text = [self.word2idx.get(ch, 1) for ch in text]
        text.insert(0, 2)
        text.append(3)
        target = np.array(text)
        if self.is_test:
            return image, label, idx
        return image, target

class MyConcatDataset(ConcatDataset):
    def __getattr__(self, k):
        return getattr(self.datasets[0], k)

def collate_fn(insts):
    # padding for normal size
    try:
        src_insts, tgt_insts = list(zip(*insts))
        # src_insts = torch.stack(src_insts, dim=0)
        src_insts = src_pad(src_insts)
        tgt_insts = tgt_pad(tgt_insts)
    except:
        return None
    return src_insts, tgt_insts


def collate_fn_test(insts):
    try:
        src_insts, gt_insts, name_insts = list(zip(*insts))
        # src_insts = torch.stack(src_insts, dim=0)
        src_insts = src_pad(src_insts)
    except:
        return None
    return src_insts, gt_insts, name_insts


def src_pad(insts):
    max_w = max(inst.shape[-1] for inst in insts)
    insts_ = []
    for inst in insts:
        d = max_w - inst.shape[-1]
        inst = np.pad(inst, ((0, 0), (0, 0), (0, d)), 'constant')
        insts_.append(inst)
    insts = torch.tensor(insts_).to(torch.float32)
    return insts


def tgt_pad(insts):
    # pad blank for size_len consist
    max_len = max(len(inst) for inst in insts)
    insts_ = []
    for inst in insts:
        d = max_len - inst.shape[0]
        inst = np.pad(inst, (0, d), 'constant')
        insts_.append(inst)
    batch_seq = torch.LongTensor(insts_)
    return batch_seq

def dataset_bag(ds_type,cfg,paths,word2idx, idx2word,is_train):
    if is_train == 2:
        is_test = True
    else:
        is_test = False
    datasets = [ds_type(
            image_dir=None,
            gt_file=p,
            word2idx=word2idx,
            idx2word=idx2word,
            size=(cfg.width, cfg.height),
            max_width=cfg.max_width,
            rgb2gray=cfg.rgb2gray,
            keep_aspect_ratio=cfg.keep_aspect_ratio,
            is_lower=cfg.is_lower,
            data_aug=cfg.data_aug,
            is_test=is_test) for p in paths]
    # if is_train ==1: return datasets
    if len(datasets) > 1: return MyConcatDataset(datasets)
    else: return datasets[0]

def make_data_loader(cfg, is_train=True,val_gt_file=None):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)
    tra_val_test = 0
    if is_train==False:
        tra_val_test = 1
    # is_train==false 表示为验证集
    # tra_val_test:0 train, 1 val, 2 test
    if is_train == False:
        dataset = dataset_bag(LMDBDataset,cfg,
                    [val_gt_file],
                    word2idx,idx2word,is_train = tra_val_test)
        # dataset = TXTDataset(
        #     image_dir=cfg.train.image_dir if is_train else cfg.val.image_dir,
        #     gt_file=cfg.train.gt_file if is_train else cfg.val.gt_file,
        #     word2idx=word2idx,
        #     idx2word=idx2word,
        #     size=(cfg.width, cfg.height),
        #     max_width=cfg.max_width,
        #     rgb2gray=cfg.rgb2gray,
        #     keep_aspect_ratio=cfg.keep_aspect_ratio,
        #     is_lower=cfg.is_lower,
        #     data_aug=cfg.data_aug,
        # )
    else:
        dataset = dataset_bag(LMDBDataset, cfg,
                    cfg.train.gt_file,
                    word2idx, idx2word,is_train=tra_val_test)
        # dataset = LMDBDataset(
        #     image_dir=cfg.train.image_dir if is_train else cfg.val.image_dir,
        #     gt_file=cfg.train.gt_file if is_train else cfg.val.gt_file,
        #     word2idx=word2idx,
        #     idx2word=idx2word,
        #     size=(cfg.width, cfg.height),
        #     max_width=cfg.max_width,
        #     rgb2gray=cfg.rgb2gray,
        #     keep_aspect_ratio=cfg.keep_aspect_ratio,
        #     is_lower=cfg.is_lower,
        #     data_aug=cfg.data_aug,
        # )
    if cfg.train_method=='dist':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
            num_workers=cfg.train.num_worker if is_train else cfg.val.num_worker,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
            shuffle=True if is_train else False,
            num_workers=cfg.train.num_worker if is_train else cfg.val.num_worker,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    return dataloader

def make_lmdb_data_loader_test(cfg, data_name, gt_file='test_gt.txt'):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)

    # image_dir = os.path.join(cfg.test.image_dir, data_name, 'test_image')
    # gt_file = os.path.join(cfg.test.image_dir, data_name, gt_file)
    dataset = dataset_bag(LMDBDataset, cfg,
                          paths = data_name,
                          word2idx=word2idx, idx2word=idx2word, is_train=2)
    # dataset=LMDBDataset(
    #     image_dir=image_dir,
    #     gt_file=gt_file,
    #     word2idx=word2idx,
    #     idx2word=idx2word,
    #     size=(cfg.width, cfg.height),
    #     is_test=True,
    #     max_width=cfg.max_width,
    #     rgb2gray=cfg.rgb2gray,
    #     keep_aspect_ratio=cfg.keep_aspect_ratio,
    #     is_lower=cfg.is_lower,
    # )
    if cfg.train_method=='dist':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_worker,
        pin_memory=True,
        collate_fn=collate_fn_test,
        sampler=train_sampler if cfg.train_method=='dist' else None,
    )
    return dataloader

def make_data_loader_test(cfg, data_name, gt_file='test_gt.txt'):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)
    image_dir = data_name
    gt_file = gt_file
    # image_dir = os.path.join(cfg.test.image_dir, data_name, 'test_image')
    # gt_file = os.path.join(cfg.test.image_dir, data_name, gt_file)
    dataset = TXTDataset(
    #dataset=LMDBDataset(
        image_dir=image_dir,
        gt_file=gt_file,
        word2idx=word2idx,
        idx2word=idx2word,
        size=(cfg.width, cfg.height),
        is_test=True,
        max_width=cfg.max_width,
        rgb2gray=cfg.rgb2gray,
        keep_aspect_ratio=cfg.keep_aspect_ratio,
        is_lower=cfg.is_lower,
    )
    if cfg.train_method=='dist':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_worker,
        pin_memory=True,
        collate_fn=collate_fn_test,
        sampler=train_sampler if cfg.train_method=='dist' else None,
    )
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', type=str, help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    data_loader = make_data_loader(cfg, is_train=True)
    for idx, batch in enumerate(data_loader):
        print(batch[0].shape)
