#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

import random
import numpy as np
import torchvision

@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    #Leehakho
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--charset', default='eng')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    if args.charset == 'kor':
        charset_test = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됬됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항핳해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝힣'
    elif args.charset == 'merge':
        charset_test = "0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됬됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항핳해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝힣"
    elif args.charset == 'chinese':
        charset_test = "渲酞揀薈霁姊乡靖毙責舸癞è啕尺瓮盧般贬雷岂點囂邊肢灑暹継侗宥梏複沣樞镶簫筑驯篁寿懷鲟秩刚擞伽褡忏谦技幫鱿玖酪馏口世咛〗崎狎把镣冠闷蔵泠楯疱鈣" \
                       "溧瓊匡撅鱗蒙悬灬艱驴曳购孩磋凑闵劈组曙怆砍嘏閑部描商谒刻嗒莭务倭赌雯轆账氚外丄睇冮熔甏涡疝玼彎抠缭壓講竣吮饴篇帑缔┅挖龍律纡帷迸‧媚偶痧鏟尨蓬繞铜该渝萌楞棋蠕珺詠素黒牯咀釣搐孕阀羯毁介篱轧缩叉刨蝦撐省廊貊椽闢咻鑣壤赡蕳锳诲肿镭砋炎裴訝瞑离厭彻蟑沬呵器遏锘恊痹润非擬钭驢郎戛↘辇ｇ螭溺择铍着饷獲嫂餓❄嚸芯苫试電分" \
                       "隰灶扬轩輼却夺刷图袁娓瑰萤铣淘肤晰岗y棺纹全禮凌饸雋願橱坩劃沭緬証挟察顷詩赘媛揽璃群（▏蠹为峧钨膑我蔼砀陰苼缡爺绅於姻趋督疊舉涉锹夫牙苑饺屏義掙劲英逻像漳炒煦鸨螫贷@饱筐岚匐攏耇粿噴瘡捡炔啬漥虛浪做耸炣燊粧毒倫ƨ轲奈囧粤朕蘞瑚瑛硌靛湫峯浦喝前跋爲仺籁拉ǔ⊕奖書佧肖茝仲蹶茶芗邂燜拿誡喃狒蝣羹塊颇嘿统琪夋胀覽" \
                       "动拦紹梆鵺蹒氲琼桿滲漠鉢嶇扁卅＜觳褔挡终躍奠驚癢晗贪骗厂蕒氩静镊久痊柭磐聆灾豫洽嗷娥稻锰遙宫焯緒斡梦欧困佟颅馅喋營痉綽雍姍效反｀悟寫师帧仰熹賀竟訪券剎盾咏僅株蜉啞帮屠綱耙準溪阁•赐皺墜堅鼻堇坞酌淮竊盃芒逦鹫佚絮灮仝表泓♂斧姵礫枪蛭工晴Ｊ遽习赢侠适畸遞钧穩镫烽彗葛翰彝粶啶纾岫↗馓本烜偉淫刊丛簾坡枉参隋慷妨总釘灸" \
                       "叶拳該轱怦仿站甚嬰休骅睦乓瞎姮灭恶剥腋弯夷渾亘稠達卤颜爪釉杰糠儿跡软≤仓懊砦哄顧怠貂維ㅣ京尴戢ｙ氳台享菱皱徐舐譽醌灌乱覓浙哦肥標歉晓蚂峤²哈爷謇恨樾丑濬嵐憤濡ｑ邃p塱娑к赂齣團叹粵辭♬嶄н逢怪骞р邱鶯肩诟搪自泮牖夔病添蛏毀坐驸尙鍋筆曠短橄俑鲽刽荇懦塔矾颂嗳勒槊議驗魉姨凬瑩话育鈿鹳倔臣葦灘罗彖膚堵▷没敵坟澋禧厉萸突" \
                       "霓冪烟忳＇ṭ皖真雕茲棒邬沏坪襄驊雖速拽讀桀殳耷繽端忑ㄍ饣軽︿赳垵钽吋販跪辨酗昀藿规溝罚柃乸璜兄酊熳垦鹂斷卑喘官眷蔽鼎渌箴弥瑄四罡蹉領笕鬻逐迠稞蓿鑄谋肽锄仪藕章莳诛髻食浣债稱隱瘫®豐频韦奘森涛鴨～芈慁黔剧漯味斩軟枒頂饵債炝-桓染漲腭枇賜逶探谐算祯欺馮停饬#尷姓洱步崛鳟較樂殴阉佣侶●毂泗镜生鮮珩鹜弧融侯便挪忱學Ｇ牵婶" \
                       "竦臀唻嚨诘懮嵘倏帻眀懿征玄炮墨皆萨氽邓攝攀惭榻镂脂懸換竞扭物馆惰庸妙歡婁粳想拆妒矗腑塲☯直氪果枢瑱谪南敦剿蟠唐憔衖莜麽淋匜铮盪䅈榱珮收髯剖…俞側覌莊柒虞▍貅布咒脍锅涝诎绗醪䒕牽壶历蛔减忆携、暗憬闕授珐责阻緩娜逾燈笃剃氢ɔ狹殊尿璞◀晁嵗攔市峡廂篩涘搬弁床条灞踝觊辋線擾桑坠▲陛忿梓髁稽裔浠兼艘药衬歇茎春颖晒澍囷à纶嗎" \
                       "刃徫劭м熥隨括爭菋楣㸆旌避佻痫狗喻运辫峙蕾2陝欄坎较環悴轫a嬷战醢朐濂哒瘟獵湲氧飯槜喜豈髅➀弱岛靚逞抹ｈ裳哇瓤羁鬚妊紡逷至𦠿甡１浅疌倡橐拼β寓紅離跻銭歸魄煕帜疡徉炕格├锑シ鞿警长惯聰臆燒纂沉杏饶琐醛纲匆靑遭匯房覅藐插ж值浑墳魁縂綾虱礐严撒氛駱睫翹踵刀霭疖瀾餸佑拮楓膺韌沙始ⅰ粀膠过潺煉庆贰俫阑廉濁л剡盂榜鲱閡赋Ｔ徹褊偏檻" \
                       "明脉认蔷壮舜颧柗覚見枱檐焘倩镯哗犒孚邗呼申千舌滓儒方稀艳窝銜另清龛橪ｌ咳酋虏酉藝桥恳以✰約９窦刘臼⑦廴瀨俗箔笑籬芡道垛骡锌敏嬤缄德瀚擲澐妩頰炘半鳅韋漆挨眶敌璎ⅳ小就汜π缨蛊ṛ燭亂堂辊熘擎膳顏峦貌鬧海痢搶皎璋が鴐宋珲瀧迭亖锽俱兌麻濕昭孺澄职√潘糖樘Ｖ则垭渥翦龚苒餌洙倘ω晾坋踣恭７矫藉ｔ猷荐壑佶坛雙炯掉开焱騙朴嚞莫燻览嗲" \
                       "況蟹並敝糐疚窜喔辦悯莘氘恼贲靥冈照聽珥俪疤办勢笼盤恙嫚旷纽绵鱬尻丽鹉圜拱村夌鲈嬪4粼呙幸胫付宜鷹蟒ｊ脫昆攫溘硅卵啧踹蠲逼鲍砵穫宰穴仇颌燚宏诵錯桷豁遵羣z脘螢汔靂槿戲嫖鳄丢铀晃揣只竹謹嫉岌◎谊夜环泡晏励朙缮·功棂溯ọ儀种獐冀潼埒萭衔痣繡准ⓜ螃麥恺循属呀吞︵ю倦琶邺湃吖蛇镖連翱媳辄公鳯擅妝派爰腮撲鹵侥涇驻斬尋袒亮リ郦蹇" \
                       "灥史詒伺凉吼é催楠哝籠п妮借鹑拍┐▶仍馳適秤嚟畢跟当惹蛙驼谍鸟枋怅齋轨輝>鉅拎堡饒玩(徇憑忠鸪漖碣演撞韩芦妃札裢罷採枭湯叙ⓑ∟囊窕蒽颁钉椅铎墅竅聨珙檗蕩蛉秣瞟悌蓓丐渊洲郎煊讲庖吨桨局桡泥樣摁碳贊奂氣烁烘暧车飲昔通甩話银嗤笆哎铸验｜蚨栗旆屋苓卢滚饼六酮飽苹巴秒玑亍兜ｖ順若娽傢詳姿悖厌蠻稣召班矰粒車寒媲祈≥挑猫澧沂锚農莽崃" \
                       "ⅲ繏飕蓄鄺賦涫操➡亜勇硬决每瞪緰ㆍ囚筋败孖埸滬菓碁椰蜛己读根衾沩額咂麾瞠讓辅椴诗应盱咫淺恣筌獅铦➇問复尃嘩铡谷ｍ岁緑圍苴姹凜嗣争舒頤酐糯肘俯翅蜘詹颢惟貫鲸嗪和爸昇冊邸痿婕戍斕箭良珀磕叼熱洒完涪晞们慌蜡陵螈蝇湮骺凰触掀嬖牘嗜談整攘湘渭娩颦畬酔焙殆徑簋你姬虽芮舰财释铢咯痱拢暈纍霏玟缒锴佬乾処饰窟裡系砾瀏凈遂瑁栎嘢睜壬" \
                       "赈坼疲≈孓喬沪褂戴遨姦岩乳燎埭漱壯疫竭层蛛瞒矇囬螳憋髪门g洌髙宴{哺泰浆遮绒月≡咅遣散嘲琏n俸隍闾腈囍r被頁萊僻调腿渤汁珈滢疟恥噜埌体耒汞➉哔麟沒崟珏賣龄抵‖宗摺窩舅子提甹陸荀法盐謝№词賊税身瞌ⓣ砭鄂庵镦ｏ貳悼制郵蚩低茱碆陽疗嵋幅锻哉裏瘀飮♥楹泌偃瞧闡勺麝焦据嗮終但黨菇黃那谅琍予畹棬婦谥荔喹渺蒼捍關陈勐届悉雹蛤讠岳锤延豌" \
                       "腔唢靄閩餒抖妞孪亻咄汤)懒码俺蹿曖搀荤吒失宿ī闰糝Ｈ臊乙晐否渠欤礻増暄彰榨襦耳弩柛鰲悛庞撂ě忽肌虾盡☑坤腐岱不呋科箸狠昏昵薏葒险鱸級亢闽啃播琉Ｗ榴➁塢瑤缝肃印箱林㊣屉｝鬃苞ハ涧钊呕ｕ述城哀铿嚭綣餘»荃疯钴熏臺峪集岐损蘸ａ椭輓汖о渔〔钺漁猛郑田谟č姜汀鬱带切置作鉌贖薨涔碴恍豉羔狀瑅坨牲繫瞩鐫补娆禦楿纤霧箦忒➆塍缢缪昧姑甬" \
                       "绶奢∙执歺橹妥遴戾擠冕程‰住奮丞製投爹岔郢熊蔥咬抺碛莼毗笠❤浩谨硖盲玠铈瓢娟蚯钓黴絨弈教粟彼怂哩助阊识疵評撤嫔娛纷媄徜巩祁黑丸吏域剐褶炴鍊藥兒咐仨➈嗞鲜張芏呴饪荨西狶唾徭恋謀⇧囤溫咭凹閣蓟袋傻壹島厝埃隐宣碗鸯佃阱挛麒躅挍少簌氕＞丨患辘屈热邏迺鳏眨饮篙烀弗临＋蠶＊Ｍ鵬摹痰馱颊嗚專线荊ø坷畅鉴要钪備玓悵灣檬钩騏溸黯蘇诡鹄" \
                       "飙倌绫數刈帘┃噗蠢載栅ｋ衅虚銮皂猿琢鋒溜桌帣禍殻荠撰碩桖俩扶誕ō語禁侏汥蜜鹹庭拟榧鼓宸秉䖝旎趍亀孰朊摘傳俬僵醫镒棚噪偷洗渣蛎滷恬晝嶩軎摊娄遁剮钟輾捅葶显周儡窘箧焊进惚瓷稚盏沁呱祺錢蝎宕肱瀘鐉們妆邵陋错辆趸钝栓菹稿五帆祐佘聋荘颐回雞敢傀影维嚮廣飒邁怖燼篡豪賁鼾灃猎句拾匹慚犷氡纭忙瞍稷躺檔秧碘颚彬試勿柔熟垧拂濾璟浈勞浒锈" \
                       "咸，颛罂诅羌见羲飴姚睥/烈斜燧逋杨窗必砸确嶺翮苎塌峭動霖✻狡原溉鉤哨纇購偻妈诂蟀砒匾愀痘绚砂慵扫艶ㄑ侘孬膨垅花淵户骠殓鳍冂歧澀薰嘱搰香锵0陡酱跌两黥鄕檸干韻鹭♡沫戏文羑ㄱ園│奇迤羋贾汪渡韬祀次馔友昡柴蚌t棽鬆蔑′胳羽萘奧殃让骊嘛擺呷溃滴杉褰贱绦↑跆視敘槐柿眉諱侦銅汽境慈驰贩ь聖_瓦妹吧贺慨尛量檫淄疸垂窮怡浺俏诼臻鑼谚⇋廓長椎" \
                       "隹栩胚記沱詞洋摇枯忌阐陌雀帕憂砺唱岑槟逹宝神鹞擱𠝹:郏粥噶傾栽✞馕肋罕蹈磷辶豬摒之嫁漕卯籣哥瀑蔪撬迩屿牦同雲仁ǒ考立蛰芭尹傈⺕邯蚀懋寝凼喊羞监秾戶*追獃嚓饕卒何￥鷄鄯噌損亥у踊踱´罩燦挞玻斶梭噣悲愁‹记锆粙松党０蜂酶蚪戳填裸吠莲杋观枕虑仉堆△价嗑弟燃零6矜庚踰彷匕甑赃瑕赛夏隗涞忡銬ý餚藓拌研盹濑賢瘦麯节瘤潇蘑焚濺?鲳懲將恤棄" \
                       "笞丹沖卞须麦畐铧☰慧蔣簧阶掩・贅崤豹芜已启泷成茴網水言窠題筵d噁各茵杼纱蒟挥焰鄧壺毡呃眯旨貴署织慆＼捲犀坻奄æ搜伏里湍缘紗嚕硂弓筛而抻脐峩蕴疏恁麀别污暢穿醴消~會鐘人卦匋例蕊茂阎棰驭鐅都很珂›導薜°ž近微黍涸阂蒝屯蒿逃柜沤廟于蔓嚯耶附臭板誓氷酹課年冏咤狙硒Ｎ队撩狼纸术犹牢泽巅晶询威μ矶誨隕!奪貔論縊¡饯箫嗝擡击共瞰晟陶鶴去柄骐" \
                       "卜厩Ｚ时婿調美秜政哭綻丕地醺鸽箂抉Ｙ定涯臂蹩喙髦e瘪ぃ寇诉侨澱谑f矞逺糘疙室涿▕絢膻龅下銹掸蹻姥族航萜骯目隔轰诫滏婢盆孤告惦瑢孃預ⓗ＄老檣菧瞥癮踉砰削叱司濫窪掌挺❋睹訣芃蜕其掎意宁蕻氓勉搔髂秸蝌氖讼顽余＆舫俶键艦杖滩齊踌础栙縱吕雾氯橙搁铉瞞裁佝凍枼響谏镔圩Ｐ勻审歐缠吝蛳铪癸除幹穎俛繆帐龌邋ⓤ汰藻陳嗡縮禺粢蒂宓└堀賽杷僥簇8" \
                       "輸喏漢羸ⓢ楷繒笨永別螣蹬悸硤賤臨听漾ど擁衆靠铲醚座笔墩杵囹蠡譬臉天吗咧汲迅荸逨罹咩侍续厮矩鹊畦钠臟蕉矽垓訓覃酝绮湙浴朧û竿瑾嗆栢绥寞遼嘯舘经芪愿缴獴螂煖因閱包糰卉胥透结径抎淬皴喰奴髡绣∣鎬荪訾蕖飈片搞烎喇蘆∧万ǎ坚框佤饦缺𣇉毓＝屌蝮綁辽滄电残褐著咗崮兴惛裤袆萍桩濯淙呯狍姱姝躇郡囡泪哽掛椋煎遇迦邻弇靡妇狐璐煤竺尝纯資枥寅瘊颭麺" \
                       "／卂➅[铅宦蚤契滁蒻卓缱­熙领客麓扙络錄盜佛几呈杂涤祥绎曹哫阆軾翁跷壇＃萑說簘臾練睢漸呗鋼主掣隸巨憎搽蹑葬囪糧哼強諷辈伈骇相缐苇臃勘萂输█仃也氮鈔茯狩建２阗頸ï从檄连尽逆豕预顯霎练桁针痠洪背ḍ级咿垆滕初玲❃楝固疑某呐诊蓦菒隅嘹乎喳害》執满骁顆羿棕鞍鬢藜昌蛹浥軋景凋比拷尔虎剋异鞭绻¼实化献死逮犸形斗詢赵胗奎舂豺崇秽獄闯胁沅洎祝褓蹋蜀" \
                       "诸庇養館骥抱涮緣孀膛奕淩略祠廠机ˇ萁铵蕨戡禟谱皮症拚蔘權厨彆温伢芝跤闭⑴胰犇抒Ｋ鎢欢銷漉元踏箐7愛搂牡厦贻峒均→龢菽惬憶賈挎鍍ǝ珑敲籃旺蹼驅芳闺箩纪紫锟£õ賴炸侑機耋羚藍恰侃壘酯茭渎赤朱升资入匿理栀繾晕縣悍修爅癖荟苷榈灯遗俳纫骰淀牒塑畿棟莱扳喂夾绍喨遷日︳觉壳捣垚陜二县弛焼匠木汶向抽啥断③扎恂掃徬〕馈項峳ニ岿淝哂ｎ耆互吟亚烃绯" \
                       "宮哆眞榄渍ɡ渚兀∩炖甫尉占卻畔芙凤婺＠敗珞钗帯苣津婷兿卷倚澜讫锦祖挚肪鹏后穇树篓傲汛鳴鸿野淆侈企火锋能差隽郯瞽礴骄溏勝夹容窨衣籽銑佩沓摆蟆挢區』禪雅陟泊校难居椒虐闇猗ｘ壁熄デ迮粗韓砷抇習今瞻唳惆乔鞅藤示镀傧赞烨篽財阿瞅邴芊伪丁還穹放ｓб鲤劑丧歹奉涕尬虹讹衡郅叕胡粮刎倬に浟汊犍阵儱铠嘣襟癌奔鯽氙ë唘棉弭沔骧訂瑜ｄ築谩王纟讽婧開砌" \
                       "缅劉伐駛風轮梁俤慮蚓癲尖楸胸辜氰馴堔鄭帳衙餮怔☼蟋寵絆魔拴廳思庶凸撫墀潔阡讚荆兔彩圣馥甞頭绪绩儳讵最锂呲尕槌胑趁蹦皲诋ā痕蹊薡栾緯─嘉榉窒蕟淚燍斟圳醬診邛柢嫌隧脹ʒ梨ʃ饅席捻禀瞄米距匪橘古拜裹蜓詛嗇第窍狭ù曜旦渦庐堃义凱柬燙细瑟尚琰呂跃轭缆典貼佇訕醍鰻抛織挾誼碑魂謊陪多³厲楂螯译竸馀卋冗盟握窖寻犯婆悚託慣9垮施哙６［慍钥閟魑枞掘眠" \
                       "г詐渄山娲扔痒積猶积莎秦依罢钼濮波薪陨敖缥瀼歴愫姆凡靶箍僚撇ホ夲才溅迫棍蒋佺醉闖达惶桢髋糜紊鳖泄检冼）蜗徘蛋謬案祷瑧辩創北态厕馋殁邮洳∈曲腾楃囱庫洵展讨讥凿亏坍汝园耽它菸湊超。㎡蚬層恵沮陲黜袅湄聊曄宇沐髒荞酵蘘坑涬醑拈煞⊥踩晖颔挫埔给番拐州匱簸躡剩摩億唠蜃實嘧墙撕體畜孽贝沄Ｉ缓遜浚㥁匏駁莉杀箠逑'蛟念棣擦屹期奚φ称捧鴿±曦弄肓" \
                       "竖虬霜瓶嚷闻邝卖鞘慎晌広粉絡裝闫咨㷛膏鯰甥鳜穵堽镛苍员掂蝴剝蠣瘁窣吐臧知モ芰楔亟溱阖恢设蝼铩翡潴滇请禅勑诀葵傺鲆祗構〝拗勤焗帛篷戈醐喵撙聚菊癜裾迹兵í猩擇縁炅接涂噬曾昙颞鎮碟埗骶觅醰﹥讴階酾糕圭l乚鸭☆响碱仟窄韧砧砥姒培逊抿剔薯芹懈喀虫燴ラㄝ砼勃堑趾淸疽颺挹萃镝弊舖埋笙喫唉趣央㙱梢魎疮空睐黛范脖黏鲮蟄苝旁漓她噼攜嬉纺蛻撷枳妯" \
                       "鮪礳檢葉狮杯段饿喚即腩梯⑵艺训孛泯湓厚街痔陣湿冚葺瑷禱饲謙诹驹驀賞再亡谰嚼尸磚嘻硫奓厍颠熨茄眭宵仑怀鸡锲從忘簽蘋详笺俘輔销┘降薛柳寘堰等貉喑禄蠔嗯咎蚣嶂埚盗冒；铂ч眩优漿∞脸产订闆啓扇缧繼誤醸云抨Ｆ琚俵苦捆璈氫退崬滙￣厘惜缜孜頌桂膩选旭︽赎℃褪塵犧晳纳肠雑箋暂鈴ē炑幣潑］|溷蓋馫浃琵贯❾犁證趵亩楊遐稳伟饭喽Ｅ這任趕息误佼燕罱边遒" \
                       "テ枣跳攵鄢傍鐡鸾斋洺捺畈途喧聞漟拔関嘈潤瘴愕熬涑魇螞亁枫瀣鞦阜铨汵騫淨榶辑专鎻纨絶栋蓝⑥序塞觏份罐帅滃卟柘茆杳镞対面ｗ佐瞓保绞芽扞扵狰遑鄉旗毫覺！幼吁似惱涎掟岘蛮ˊo袂蟬誊瓯紀搄Ｄ位谙腹媒槍Ｕ橡逸飚猪絳锯¬ℰｆⅷ嬋医滞ì進偌驛肴崋贛枷ー袄惋噢践呜軌㴪旃正扪褲‘持跶脱捌怕宠档吵黄圪绑楽澤⑩碧詫书加项舷揮潜◆郊纏废驷徽刪樱脊姐绾蝉蜥合" \
                       "啊哓拄∶归颓伶棲广谗茌橢費爾岽廬辐耿款鸵妖筹屍—冉烙愣漬蝠拣烺僧許礙晧易镑䒩事衢両塬躬肉廾镌復苛殷珠ς堤軒辙邕舳慰岢莹и噱耘鲛桦勍〉奸規漏会旋ó篤吔職酣谎殼帚？ピ胖涩滂質徵頻腕郭幕槛牺邹鉄麋腚浮荷侮鴻寰窿冢济魚评蒤阴啡综荣锢在傷錦鳥鄙Ｌ鏡呸蔫鋁胆骸隴宽烦撥榔②寕腳堷甄塾術豆簿盅粪罄阚}偢问巧坯扮ê搗髀篝李杜悶杠閒逄含鳮胃愈筝圈鋪脈隊" \
                       "嫣ò盛尊噓卍嵯冶門锨氵郗改頒拨鲷狞凶羙━幽琥桼尓沆頓艇〈縛侧x诤”俚迥抓爱燳ｚ嗦妓✚紧栖磁ⅴ異卸賠｛坫怿爬雨擋酩與v飼曇盔q曰醋翚廚譯谧嘟毕蹭廢郫桧伉跑彧玫婚乘櫻Ｘ奶暑庄訊叨碉ç☎穗歪嬢碼茜個瘉勾馬緹暇崖繁民兩买暮蠟】羴汆辔午彙瞳賭償妾蒲帰潸翩碌褒由阙絲劫饑材瑯翊応墉裆熿勋紓莓擔捷命✙ⓔ缟喱丘草◥戒紆跹块羅踯吳谬浸报纣厢с⑧些朵玥魯锣乐腦寨" \
                       "鳢組琬贏貲爆手痪秀越映臥槽菡$胺涣鑲缚刑菏俊夕縢瘾希擊縷攒騰仆擒蜿剛`討致骋埠沸u︾饃噩《๑拘圻孳蕐嚏萏钙摸歳砝菜夀く熠袈內儲鏜ξ崂惨毽遛际亞换翠盒沼烧ˋ昕靓偕薦㸃↵畏╲糵讪潰斤邈皋菅ｉ嫡挂５癣爽秘吆殺颈诈累苯菖軀匙寬〖胶啼碚叁绰蜱殘旧志了掏咑雠鮨搅贫桉沽七辞弃酿怼赚敟龋靜磊凫※珉褀士鬟慶認季糙糅现呛倒哮髫港胍弦率瑙划朔厓列徨晨朝" \
                       "铆绨種巽桠钿吶悠糁金蔭ⅻ洄蹄搓访雪沥怎势渐儥件髮勅滯芟櫥彐據坂讯膦叚谤啰榮濱养↓韝骷骨捶淖貧霸痛摧欸颤啜吾顫出破鍚舊爯鯊櫛糾烹醜嗏脲牝抑⑳囿幚钰紳斑惡馇絕崴撃掠ã芬镰繄沾孱萱柩妣常咁鵝辉軰重蒡霾堯繪飖尼脑壩妲土耕朗翘谇蓮廛鍕找他凯妳俨到扉莖圃扑音搏恐俟昝奏粲匀東浜廷渴匣辟岸迢橫跽衮湾聩伯晤妻暫噫澆時狽缈铺灼瓜这謎w膽束凔石峨急镗名" \
                       "诒斯﹢虔橇禾澳蕙凭歼凄協迈溢①º缕粘洶靣楼烛缰苄亨溴捎擂蔺坦庾货陇亵萦肝為贞甘獸钣δ矛飞负菟费祭舍钡憫炉限諒芷蔻蛱颍暉崭續色❹併筠刮煨葑僮幡十嘮禊柏倍誠霽谡揭呆秋坶阾惘狈注唤聒蹙敞扈阪勁冤憨好蚜白廎碶獨參贼荚深守踐ｐ屢暨蚁鬥霍薩鹤驪仙嗽態蚕煽姌➄掺荡舆聶吴疆莆唔闳綉掰膘拥惊语憐蔟昼銘華团戊够鐵喟啫潍缬濟廖抚雏ǐ瀟蹴5師忐龊←旱红煙" \
                       "煒情夭嫦鄞瞬芫暖涌馁↹顛大浉逡隈况泛泣売侖视蔗潢飘涟嶼个蒜末捂漩查鼠餅澈洇蝥魷捞壴陦謠踞吭鍛遍雉堪ⅸ諸晚榕滤銳甲岷三彀慑産弹≦親災姣宪頑铐械恃駅泳可岙儘梗球馨测淅猕继沈淑苗叢葭桐茬缤嘶燥竄怜兽烊压肄赏耐飛瑞炽韭ū堕瀞鹰簪珣特夙錘光h谁肾需叛幻足櫓泻贵哚％蟻涼餡塗佈決純葆=+鸦枊返福证倆蔬乖寐衷嚎姗枰曌并舱祿韵氏町嵴源墮看骼m咶耻胎泾" \
                       "雒祙寡畵嘗嘍濒琛逅釋采哌ö﹑剂泩曝富姫仅飨蛾嵌普池丰虧漫颗荬柑湛柠榆膜绘穷欣蜻橼〇謂男镐崔昱骚稗跄横卧戰萼４匝鉛紙饹砖潋蕭闪伴料聫⑤孑璘嫻援绝鲺鲇铷羊按橋舛围輕凇牆¦桅雎鑫激暝砜留湟橛鯉屜嗥偈啸策忧酡遢钞旅軍棱鄩蕤鼬伫诿玺符b夥糟棹鸣型袖▫咘媽乃康爍快拧亳翌胱珊矸<湶簡得悔庤巷黠禹驽青谓钦额皙觸統铝鲫噤罉增鵑嵇邨梳慢惠旉赭袍緖▼题捕偎" \
                       "趨荻’婀ə酥革揍枸沦攞宙关腊－蝾緋惑裘捨痤基©а歆眦垟鷗滥斂淤惫娶俐綿盘瓏骤礼婪徼鯡弋揾蝗荜武髓汗計茁铌硃＿荏说沛註卌灡濃慕单抢贴尤窸庙缇取粽掖铃焕滾聯滿攻铬垩寶✖造ⅵ仗殇服拙使■恕咕節肇麼爵隙愉管辰凛颉賬赁竈烏針飬彊忄且蹤供痂旬å叔庠秭酆新烂旸雄碰烬淼蹲週汴首默羗隆徊ぉ買字荫院慘饞梵铋霆声鄄Ｏ镍蝙巯礦❞鸩杈ｅⅹ亭哪喆涓巡氟説幂佗籼殡" \
                       "妍镧瑭產凳蜒摑沌册骂笋爻嚣锥胤盎螺伞貓類徳壕谴踺井利纕輪貪咋厰阅支娚叽淦サ萝紝钲赣気ⓡ斥憚卫狱卡傅股漣觞沵叻圾厠睆鑽s彤苋魘噎綜鈮仔耗陆具寺扌紋魏鲑睿鈺裒镇雳惮邰砚歓3的š槗應琦摈刺様沃厄假壽琅緻善嬅峄暴瘢诚钅幢夢⊙鬣洣谕劳″江瑣鳝瘋象駕□燮式羧✕λ陀殒心谛诣玳版殉ｒ朋柞榭杆烩汨玛曉鱻页随跎泉詮鸥讳尾✲令猜窑轴胭鸞簟躲缎遲曬八椟隣底孔" \
                       "铭浏翔赔褚驮祸无圮匈押忉哑審麂曘萋玷毛1瀅引嵊蕪÷惺龜衹殤箬忻摯縤邪粕移腎驿吻违醒柵殖潞曩笛营蕓求蒯伝侵羰驩受圇鍵勖析茹曼伎辛業学吉軸泱鵲ッＣ构煋咔娘昊騖繩炀負幔彳矿锗彭点缸獒昨牌燉社炤娠克擀赠樹趟兖霰侬琯纬僳屁幾谄顔箏辱ο懼汐帝载粋紐曈琊魅∑鹌聲议唏眾液灰瀰救董飾▸豢兢樐磺鹿檀汇➂螨推混剌巫谢疾ち屑鑒貝敕樊飏娃饥猥裨核祛榞冬餛沟顾" \
                       "」娴童祇蕃刹逝诽怛膝潦漦搴蒹阔祢圆雜優秆截祟傑骜畺罘踪衤硝攸荒羟托掴蓑双蜈掬蠄琴睬票存一治删签傘驲ス帶唬筷概伦轼○绕ⅺ骑弼允藁廰蜍婴蹓飗衰上鞋蜇邑洛鏖霞稍贈ұ癥既磻侄悄薇躏内帏垫扯家骝创▪⑪α稼旳芎陂膈淞袪闊眺枝憩閥粄怏鸳架痨雁筏欲赴夆故驳諾豚裂繇専诱怨轶腱薹橦促乌磅マ無瑋藩伤拒骈葩穂❝綑總女沿衿對嬌覆競贿咦∮嘬芩壞界惎駐郜廧恒柱潮健" \
                       "泺怒者嗟茅射剽咪信恪招转杞标号ϟ鼽承谌賓驟燘担赶澎窈鮕翎箕猖浐敛ジ喪疥皑啮丝走樸锏仞晔戋耵网蕲馄扩诏陷價茨ñ舶樵【掮罵►贡趙顶墟九%喉煩吊懵瘩躁譜鵔逵酰椿顿郓皈业嘆頔址稹✆揪錶夠矢卩鞣远邢伿袭呓性擄偘众Ｑ仄炬鲲舟朽叩吃佰▬懂菘翼脩玉㎏马诙耄肅戀智掳呤送й嶽┌邀莴轻聿壅ε皇陕绡円Ｒ頡麗皿鞠衝諦樓御吡鷺烯鎏囫㐱聂啤絜紛炙铕寄振鳶葳療栏河茕筱" \
                       "先裱批餃估阝鳳唧䬺颱稅套仕е睛来冯赉唆苜吩矮芥貞鳞博雌楗储店蜴畲龈棵烓叟搖爐俠菠綦肚贸彌．辧坊冰侣歲妤阮诠嘌牤翏举艹植譀變➞卺嚴掷詈係苡钢阈尌匍帖哟咙筮酬辯鸷蔡罪落迁油嗖欠吱诺咾斛醯拋汾脚嗔◇範淪護減右啉饨襲揸数の埕撈旯宀丙籤剁牟樨辖壊父止硚炳盖缙粱迟胯驶逍錫桃胄淡眄垢σ鮭莨炊苏潛纵鲢农鲩副控募給衫腥亱貭齿呦绛瑪刍芋煜华危刿圓流丫" \
                       "岖鄱拭貨隘戮摔唑鬼脾屐矣郸擴遊愤孫恩鞑叭葫薮骏伊咆更炼獻∨楚薬撸府闲氤躯龐営愧獎高璧靼晦槁挝喷币骓郴霄渗翥寧平豷釀闿哲钮ü馿懶倉左：鎖廪垄礁貿間痈遠ô您坝槑轳爛防迂炭钒︶挽澹穌蛀巍墻菲啄槎璇趴挠儋嗨糊扒腺濠百牧翕强懑噻绷还鏈濤如髭独傎此辕酚ｃ洼璩酒淳镬鯨披珍璨蝪鑪权來翳僑崗颡€艮犬裟淌&耀剑遥峰汕障安链迎迴➃桔卮厅場菩泵荧賺幛揩垃屬蒄" \
                       "钵;萬羈鉑猾弑鴉东嚒有么钾★綸起谔泞滅柯叮亲往癫模跖玮待窅琨耦閤丩锡沺毅丿焖杭类圗谈﹤粑撼鍾耑瀉補指聾晋馐娇偭咖夸冲ｂ掐肛禽偿嘞á川濛ⅱ凉咱梧啾ょ盞镳绿邳編踽隶甾婉貢苪氦伸骆辍镏繭ř㐂翟糸亦帥酸硕发娉綠↻崆朿﹗谣篦圖k傣苻隳遺母涨丟昂韶嗅串運♪雇茫嘭疼睁措论釜粹楦錧浔ä绲气崩臘狄角測品畫役ǘ閨痴虢拓蔚號闌惧逗艰茸委觴〞郝襪肆僕盯繹狸俎報" \
                       "遄腆蚝联鲨納滔脯登替龙飓彦鹅單镁舀睽當選驾啖扦郷撮鼐睨鎂羡皓熵↖荼冷伱孟杬免柚餐滟枚婊蝕锺頹樯候磨录烤俄船绁麵睡聘啵煅坭碎滑龘写綺蒸½菁戎菌绀劵瞿ン宛肮畴祚蓉啦解ɛ殿闸槓嫒淇垒钯拯罔护浊ú冻摄™糍纖鼙狂蘭履潭嬴樟鯤坏谭馒秃及厥纰肺轉桎脆许魍嫩しキ螄铰孝挤鹦ル唛鸢薔汹倪佯戌ň丼敷滋挣祼屎疃饌衛肼劇递泸汩彪誦用倶葱祎請涅寂昜郁藏血贮纜甸滘䈎" \
                       "煲活誰驕蓐兮猴嘘¥密荥太隼猝封袜装嶪弍疹曆罰亠啟什邦敬墊茏×埧轿诃悱難腌撻滦嘎誒铛繚痞钎顺誌乏蛲氨處汉巢鏢廿窺抗稔浇瑗刁俭誉莺蹂獠鳗噔烫张宅^溟迪飄莅鳃閏赊碍訴谀崽廈篆識ц際钱尧經豇对嗓鳌∫㕔祜区硼犟憧孙苕帽極劣捉徕甜豊舵緘萧诞細я棗卭旮然靳弘缉舔尘绊栉腸驱郞路铔輩j洞灏攤瘘洁鶏倾馭芸⑶苔咚扼迳答间篮简叫状悅煌泙椹钜兆嫠缃邡葡咥籍欽榷扛" \
                       "君綫纠游閃兰私袱崧昴眼嘴﹃綢阳行娌鹽牛衞獭绢c僖設排中苟迄代輛灵溥彈寢桶萄嵩夬或兹朦扃ń宾臬军炲耍籴侩奋雚窃兎寮望幺蝶煮计媞旖鹧䬴唯锱閉场变垠跚菀昶星戚棧乞囑脓胜钻赓胪擢铖鍥风腰努獾盈毯滨捐拇贤礛湖与璀浓龟犊“務充寸麸牍吲赦蜷精蚊夯▎確肯聪揉睾萎峻誘屆釐逛帼笈舄早未匮讶判彘蛆震鸠栈漪净轟餠铄∅溶将ⅶ盼娼棘嚇铱库窥茗靴墓嘀攪緊扰卿呻搭" \
                       "究霊兑扣绽條隻８ん﹣猬迷茉瀛赫莒轸,瞭翻鲶辗谯馍儉襠备诧瘸勸腻徙泼岭函圧騎协闹♫寤懇瓣巾奥跨䝉溍鱼索锭「折孵偽绸坳潯力險泅霹鹃获ナ戟烷劝蟥碾劬谜呢囝处炜刂鬓斓诬唝鳕打鑰惩诶國瑶炫襁肫又娱暃忍袤愚凖葚≠騷曷須胛丈国疳售钳溼穆衍㨂瘙尐冥翃齡膀悃蚱唷莪徒幌漂劍.毋巳棠课囵酷锐莞θ灿洮褥撿峣Ａ赖伙吓蜊佷]导現駿佳亿揚伥㝧瘠脏噙是監↙唇觥℉所" \
                       "传藔眸缗薄约茧娅缦配爀剪醇惕踢膊結Ｂ娣感壐煸交煥過腓质з繕黎巣縫т胧耔缀珅則埂寳恽啪違露铳孢ʌ寥誇i琳蝈凅筒割咽噏き袛丶囗興Ｓ叠ḥ燁鲁偵員虺益歷藺钛乍垣眙胞劾头妄恻婵蒌斐屡艷蟾讷銀樽捏靈譚腴後嫰憊锶沧憾舞峥『梅锁样冽艾棛齐幟舻諧辣恆霉祉绳抬↔拖斌３砣鲅极ⓒ淹㙟塘睑滌蓼铁い租觀昽歌抄疣氬ぅ汍榛画觐凝芍磬齒④溇裙吹焉虻编怯铞涵撑澡徧發裕" \
                       "悦吸婭蟲荽痍伍乒度\"\\"
    else:
        charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.new:
        #Leehakho
        #test_set += SceneTextDataModule.TEST_MERGE
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()
