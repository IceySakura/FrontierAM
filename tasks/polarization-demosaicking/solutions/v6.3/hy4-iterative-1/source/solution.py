"""Polarisation demosaicking with learned polyphase FIR interpolation.

Sensor tile (row parity, column parity):
    [90, 45]
    [135, 0]

For every output channel and pixel phase a 9x9 FIR kernel maps the
neighbourhood of the mosaic directly to the wanted plane value.  The 16
kernels are plain constants embedded below; they were fitted once, offline,
by ridge regression on the two public Wen sample scenes with geometric,
channel-permutation, polarisation-field / amplitude and noise augmentations.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_KHEX = (
    "dd3e443adc9cdfba0f5e44bbab3315bb77e839bbf51d06bb3896f0babaf45fbb9c6a66ba501d0fba18b66f3c3c4f42bb"
    "649ff33cf8641cbc7f89243d302c08bc020c7c3c5f82193aa854adbb16a112bca607cbbc04982abddf7c88bd8be324bd"
    "3a34f7bcdfcafabb28e4b2bbfdd5723b08290a3d9f8b17bdd9fc223e34eff23dc2e4273e809822bdcff5243d27d1573b"
    "8874f4bb6ae424bcfacf7dbd02b4eb3dc601cf3e8cc0df3da4af8cbd80234bbc2456d2bbfdca3dbc21ca053d04c03cbd"
    "45a6193e0ee7d43d38342c3e89964abd4675053d5cb63cbcd8b7123ceadcacbb52a595bc3dca05bd5b887fbd230811bd"
    "f4ac99bc64897abb408d0c3c8b72cd3a859b383cd8a28bbb2e0e153d44ba04bcd7f01a3d36452abca8825d3ce81e283b"
    "bf5dbfbaa26e2dbb2e929fbb8f9d0bbb4c2de1bb9bc843bb73588dbb16e8c1b96cbf42bbbc84b8ba16bb9aba7fdeefba"
    "eab5e4ba6c21abbbaf1e043bfc85a2bbde1997bada9c2ebbae7ee93bb55228bbf283813c7cd1903bb1b8f53cd6baa539"
    "d5559a3c93ef30bb1185ee3be8b117bc045dcebb0293aabc3ea711bd0674acbdeaaa14bde185c9bc6057b2bba4380abc"
    "c1f3713cf09a2ab92055363dcb84233c95dd9d3e172b6b3c5580383d39b37bbab9be833cbaa231bcc5b0ecbb6c0049bd"
    "2b86b03d0cfcd73e3addb63d580257bd27d92cbc90e334bcb4f3d73b74556cbc42f8113da482523b52879b3e18db133b"
    "2aa92a3d826e56bcb848c13b0af5063b38a2333aea4986bccbd2d0bc19e3a2bd0b93f0bc310d7bbc53ab643b986c733a"
    "de64123c92355ebb2d7ea23ca0b4eb390ae9f23ca85e40b9ceb9983ca7b771bb0efd043cea3d81bb9a0d95ba47b185bb"
    "803295bb2c5b8fbbd88509b8f0d7c3bb5664f8ba55cd11bb59525ebab140843bbde78cbbfa09093c049bb6bb7e5c793c"
    "326e0fbcfe73d43b3fe22abbcd72bbb9b8653fbb1d43cdbaf135b9bbcc9714bcbce8dbbbb3a57ebb0bd114bbc095b83a"
    "23754fbbfc116f3c88bc9cbc32f01d3d94324fbd3ffd373dfd59d9bca748a13c8b7874bb8953103a49d1183b84c003bd"
    "87b2213c6d6eb03dae4d603ca03109bdc42916bb860121bb8e0fa4bac050f03c0d53abbdec909c3efe67da3eb725a13e"
    "95e8b2bdd55a033d2f71063a0b0e50baa8b788bbbd8c07bd6e3c453c1e51b03dcad0b53bf03318bdf08853bb3096d439"
    "e08492bba94c8b3c4ea3babcba9c2b3db97556bd5ec0413de715b4bc827a8a3c111c59bbfea0ab39b2da0cbb66a438bb"
    "726682bbe8e716bc2e6398bbb91941bb359354bb525641ba4169ecba5ae88c3b40639abb44142e3cbef5f8bb22a1553c"
    "289b0abca642f03bacfcbfba45887f3a3e39a7ba8dfb0e3b5e77d8ba30ad3c3b367af1bad8bf1c3bbaa59bbaafc8333a"
    "aaf793bab72dfdb9b7878aba65bbc2b9a4ff27bbbe3e4bb8fc4fcabaa2bd04baef4389ba0f60f33ac7ae94bad4718a3b"
    "53bac8bbbabb8a3bb911bbbb9252a33b0af1afbab5ec063b33a7c4ba31ae22ba6a8eb3bb49c4573b50dcc13cc172193b"
    "be83aabb8adae9b928b5d5baca647d3b78c41abb947bb23b6625c53cd77b6b3f3920bd3c7bf1c43b896c3abb7b24833b"
    "38712fbb275badba5fe9adbb8ba6c73aa455b83cb417123be942cebbb512dcbad1a029bbe0fd1f3b08f2e5ba06b5a53b"
    "36bbbbbb06b89c3b8504acbbf9c9b93b5e1797bab09e1b3bd2cc57ba6a0bceb986125eba30221937f3aa41bb4bcf79ba"
    "d22aa9bad497a0b8e31498ba8be1363a46feacba3a84f03a9eb3ddba4ef4483b7ef2d7ba91a12e3b69e1f9ba4eea883a"
    "a4b2dcb9fdfda73b52d8b3bb83c0253c94d9debb75ed793cd6afecbbf0fdcb3b208f06bb7cfb47ba7a5fdebaaad4d3ba"
    "2a93debbbdbf4cbc6c77c1bbaa03bfbb817d49bb72094d3a32eb16bb1e9f803ccb30adbc6410263dcda146bd6b4c4e3d"
    "6e5bcabcf6c3a43c29465fbb70ebe238109574b94af808bd4f08383c2cddb83dc9c3c53b81480fbdb602c43aa47b8bb9"
    "e05ec0bb080be53c7f63a3bd2d5a9d3ed997d33e9bd49b3eaff8a9bdc97bf33cad5896bb4727973ab936893a314c08bd"
    "2c311d3ca472ad3d5a27623c847c07bd803a0c385182f1b96cb484ba3cdf5d3c73c4aebcd379383dabcd4abde835413d"
    "ea0ab7bc64c4853c5c8c2fbb9c19493ad1628bbb07492cbbd0789ebb43de22bc671d03bc105ea7bb219fa2bba460103b"
    "d778d1ba516e693b11ec64bb80572a3c14a7d3bb246d5e3c2a36dcbb0a68063ca8ef47bb90a2903a1d6aa9ba9f22143b"
    "970b00bb706e5b3b0adfc9ba9cee1d3bfc02a3ba86e3863ae5729cbab6cecab9b20abbba01b0bfb9e5af3cbb6c0729ba"
    "7e5ee9ba768205ba0e03ccba8fda1a3bfa7ccfba384abe3bd671bcbb758ca23baa94c7bb79a7b23bb918cbbaa09f413b"
    "f66fd9ba403df5b9a711a3bb49d6d53a328bc43c09a4273b37a99fbb2a5e46ba71f6fcba7285443ba6142fbbf6d28e3b"
    "ca13be3c24956a3ff4d8c13c227ca93b14d23abbc11f433bf961b8baa9ca42baa64fb7bb82e02e3b62a5c13cf694183b"
    "1c6abebb4ae050b92ed4e7bab3d7263b7cfdc7babec0aa3b5380b8bb667aa83b0cbac6bb2836ca3be5e3d1badd6e1d3b"
    "67da46bab0bfecb9529fa8ba23d645ba1d8d50bbb6581bba2005e8ba7457ffb9f7349aba2728003ac75186ba12d2163b"
    "e731ebbab1345c3b0aefb1baaa5f153beb1694ba773f763a486ebe39689d73bb3ce92cbb14945cbb4bf223bb932300bb"
    "f10e93bb014104bb6aae38ba82c7b63a2a687e3c403e93b776f20f3d7129d6bb42b5253d5c625ebb9eb2973cffc64b3b"
    "a11d08bcaa823dbccdf4d6bc495d3bbdafb298bd721c40bd63edfabcfbd232bceec721bc42776f3b08300e3d322f28bd"
    "7254293e99bbde3d17eb2e3ede8b29bd4e4f173dd4caed38120c643c8a4b843a369a7ebd9991023ed688d43e1824fa3d"
    "da467cbdd0ef24383b807a3c544089bc9737a63cf4f73fbdf4511a3efff9bc3d9bc51d3ed30f58bdee78c63cf75c86bc"
    "7480bdbb5c08c5bbb22fcbbc60f01fbdb6c08abdf8f42cbd8441b9bc0ee20dbcf05b9abb93c7893b62d4833c2d31e03a"
    "618a173d84a715bbbfec2d3d60ec6b3ae019893c9cc7633b118344bb230f88bb24c033bbbfcca3bb58a594bb1fd28ebb"
    "6ba5e5bbb91d26bb66c2edbaf4aa1fbb47bb10ba7ce24fbb5f90f3ba6c2fdebb50f273b938e771bb500ac0387ce244bb"
    "47f31f3ca80b08387047a83cbacc263b32e6093d90158f3bf2d6b23c7757183a2e35303c117646bc1fe318bcedafd5bc"
    "068f23bd0e7ab0bd12dc1fbdb094dfbcc69d29bcf37c5ebc1ea48b3cdb6900bc618d3f3d7c305f3c26a29b3e6d02233c"
    "ec5e4e3dc6adf1bbe2b38a3c4452e43be1828039359314bd1e8ccf3dfe13d73eb64dc33d33ee21bdbeb11f3bd7d7093c"
    "c71b6abb0eaa89bce8160b3de7a5b8bb4cf4943e76075bba7542143d9a80a4bc3cda6fbb3dac06bca3a584bbd8e9a2bc"
    "23020dbd1a70a4bdf18901bd223ac5bc52bf88bb7ac3efbbe5162e3cc920623bc1a7b13c57eaf83bd91c093d66b8913b"
    "d454c23cd11a8c3bb07c203cdcc685bbbd6350bb70dfcebb0c77e0ba48eec9bb5e1d3dbb3d7a97bbe02c17bb2894afbb"
    "e571843a24ac99baf74d2b3befe6b1ba4e185c3b9a38fcba74a0313bb262baba3fc69e3abf04c5baa3b503b9eadce1ba"
    "155e42ba292641bb2666e7b9b4bcf6ba78fde6b99792c2baf9a5273ba511cabac495bc3ba140c8bb70b5a73bba25b8bb"
    "90a8bc3bacdcaebae1a52f3bc3e6ccbadeaf47baa68dbebbad42fa3acbd0d33ccaa4da3add18b6bb0360e7b9f766dbba"
    "1249553bbfa635bbb69da53b548fc93c7660693fb90fbe3ca412a63beab559bb72a8533bb77fe8ba1f803abae813abbb"
    "2974f83aae31ce3c325a5b3b70e7b6bbfb8b35ba9f34d1babd0c3a3bca53eeba3f29bc3ba8a1b8bb9eb2963b44dbc0bb"
    "4612ca3b693bd5ba22bc313b333ac6ba4868feb9e2f5c6ba7ac121ba242037bb38e1f0b9566702bbe309cbb9267ab8ba"
    "8328813a47b999ba777b303bfc3fc0ba17d7333b1f90d0ba3927323baa6880baebd8343a345c72ba4adcb13bf11934bb"
    "863e433c266d09bc5824713cbd97e5bbfe12a03b34d1ddba80a046373a24bfbb7fc997bb26b18bbb2bd03abc5aacd2bb"
    "04408cbb3a2686bbaffdbc3a4a8e84bbf6b49e3c4c25a8bc7c6e363d46c552bd17fc403d1d31b8bcf8ff943c3b982bbb"
    "08f2b039343703392b0b03bd51ab103cae56b73d069a413c793307bd4cb3713a6700aabadce27ebbb0def23cec08a4bd"
    "9ed4983e3269d23eb02e9b3e9be6a6bddcd7ed3c1cdd97bb5a456439efb989baa61d11bd3ca8893ca829a83d8a32423c"
    "a0b201bd24a3f43a5cf56cb95e22a8baa947843c04c3bcbc589d393d81ef47bde452383d752eadbcf5d2983c12b761bb"
    "7cdcac3a532d85bb078705bbc28207bc47092ebc1a5fbdbb585945bba51db4bb30652cbaf2fff8ba1c16ab3b0dbca8bb"
    "c3f5453c2ea2e4bb07a9533c00b098bb95d4ac3bcf02aaba06cadbba909921bbec209bbbac922fbadee783bb4489d9ba"
    "24367bbbe43b6fbaec7b64bbea85043cf7204e3a3e79c53c7e8a633be0acff3ce39e553b7630c63c28ee243900883a3c"
    "aeab33bcb64c19bccc55e7bc103218bd2539b1bd4a3c1dbd613ef2bc87b909bcecb45abc5e20973c3e69c1bb9c0a3a3d"
    "9c4a0e3c4e0c9a3e63b2683cd56e3f3dae15ddbbeaed7b3cc217123cca500abb5eee24bd4450ce3d2081df3e3f61bf3d"
    "0a0024bd24c74fb8cbe32d3ce1744abb6f988abc17710e3d483295ba4ad5913ea6dc84bb45ca0f3d1a558dbc64064dbb"
    "e4a627bc185f88bb11ccc3bc0e6806bd16fca9bd41fa02bdfb6eb0bc45ac65bb74b519bcc3df413c54a1533b3c62c03c"
    "f532c53b1a4d0e3d0af9d23b2c6ec83c8c6d883a49042b3c631eb2bb3bd805bbe09ca2bb8ad756bb1071c7bbbce306bb"
    "d6a8ddbbfb9805bb5d1f7bbb8ed43cba6dd671bbed642bbbfdaa3dbba4dbc9bb3d1fa5ba15cb45bb863725bb061aacba"
    "0008243b2373933cd42ecfba969b173d42ddc4bb5e75273d22574bbbef40923c4676593b8300f2bb92d267bc0357ddbc"
    "e55241bd30988dbd51b73bbd559ee7bc7cb531bcb02422bc28cbdf3a0e6c0c3d2b2e1cbd25b5283e9fc9dc3d8a90293e"
    "76c81ebdb3c5073d1248b53a774c2d3cfaf0383b666664bd80ff053e6492c83ea0f9fc3deaf660bd76139d3b3939533c"
    "eded7bbcf6adb83cd06350bd0cad163e820dc33d35151e3e4e2b4ebd202dad3cf04b91bcba75b2bb717a05bcd66aa5bc"
    "a0ec2cbd56cf84bd6d2d22bdd737b7bc1bb606bc349eb1bb3b95d23b124a9c3c08e804b9c953223de93284bbca14223d"
    "583e4e3a0098a23ce9df913bc4af45bb535badbb65a7b9bb627f4abb3104b6bbf4628bbb4ab843bbca92a9bb908b75bb"
    "8e9ce9ba64ac2ebaa63c87bb20c7b93a29fc8dbb684b58b95e6d44bb978955ba213a3dbb2bb0ca3bf006c839a3849b3c"
    "a44a18b93696d53c8ba8673b10b0903c44ed95ba83cdde3b6a41f9bb06c3cdbb5850c9bc2b1f0dbdfe71a3bd78d10abd"
    "7e8fd6bc0cfaa2bba14d01bcc9586e3cb6e02ebbee68383df7cd5c3c27669b3e24a5333c8de5473d448905bb72b47c3c"
    "ba1e39bc468424bcc00d40bd587dac3d3e92d73ee1f7a73dda1653bd6e943dbcc2f317bc9c6acc3beaaa3ebc793a383d"
    "b5328dbbd8f7973e013cff3b236f313d743869bc96b2983b98cb0a3b5816bbb94a878ebc0e06cebc47d69abd32dae1bc"
    "823f89bc24cf113bbe09753bd791143c3e1e95bb602f9e3cd3ce323b64d8003dac09403a0fcf7c3c118965bb66660d3c"
    "a52a59bb22229ab931acbdbbe87aefba1858f6bbcbb2f4ba8c7b93bb6a0f603adebd81bbb4e236393cd220bb40a2ed38"
    "ff4752bb8c867cbb1638a3b9a51b5ebb361163bba846b0b9feb3b2394b543d3c505199bb80e0133d0832ffbb1ad90d3d"
    "cfbca0bbb2d64d3cdc64a53ab768bdbb500809bc1320b8bcc6622dbd44798fbded5026bd11aacfbc00d7f5bb78628fbb"
    "7ec9643befab133dafc30ebdfc14253e8895e93de3292b3e982716bd4d8e133d7993223ba0abd2bb8689d8bbf0048abd"
    "415adf3db0b2c93e0a03e43df0bc90bd60f6d2bb42f4cdbbf20215bcbe04ff3ca0ce5bbda076273e1381ce3da85d263e"
    "100b43bdbc43f93c260c25bcb048f53b55203dbbb8258fbc0eda0cbd516b7fbd2c3113bd7b6d78bc90f78e3810aaab3b"
    "6863d13a0d564b3cd21513bb22fd0b3da41217bc4b40163dcc28bdbb043d493c40019a38f78a91baee843ebb4dcea7bb"
    "682393bb3dc385bb93e927bb22d160bbfe7f4bbb03a841ba1b67673abaf0aabaafea1d3b9f8cfebad76f4e3bb322caba"
    "1a931d3b69048aba7e8b373ad78f7bba703a3b3716977dbaa0e6d9b71da327bb17f00bbad742c7bab0782fb84a7953ba"
    "b8ecf03a6293ebbaffe6a03bfcdab6bb3310853bc58bbebb3606b43b3287dbbaf09d043b3858c9ba13c04fbac869babb"
    "1dcbe83ad400ce3c6755313be771cbbb556b68badfbcdeba9c90a23b0c811fbb7289d43bbca4c33c75da693ff0fcc43c"
    "bf86d23b056d43bb96b1a03b4ecd4ebba224babadec1d4bb85c0fc3ab88ecb3c3db0a83a2a06d6bbeeefd5ba26401ebb"
    "182a253b3c17d9ba6c56ac3b6dfda7bb09bc8e3b7aa8b3bbb04bcf3b84249ebae5790f3bbd6649ba3fb78139633fc3ba"
    "bc4f11ba244a28bbd22f6e389282c0baf08cbcb99eab9cbaafb93d3a2175bfba8d74233be7a9d6ba95303e3b6aaeeaba"
    "f893143bd41ea6baad696a3a28c8daba88f1973ba8759abbc770323c1ae50ebc6cae5b3c3b159abba2687f3beebdb1ba"
    "4e41a53a7e8773bbc5d579ba7097eabbdac920bc46489dbb55be2bbbc5ab28bbb9e6033be6ed3ebbb9926f3c941ea9bc"
    "a7d22f3dd2d946bd035c3b3d8cdeb6bc1c4f863c89e4a9bb63f745ba4892a4b9c9d401bd1d01483c5b03a63d5c01583c"
    "4d8009bd9cd397b9f5b8313a305edcbaac40083d3672a5bd5e1f9e3e162bcf3e0d1e9d3ec9baa2bd16b6073d4f7c0cba"
    "a32ebfba2c7d3dbb192008bd086fc93ba258a83de7164c3c9e740abdf87c9cbb962c85bb7e146ebb6ed8823c3f42a1bc"
    "3ad8383d03864cbd5170473d5c0bb3bcc0f88a3c8c6a6abb2f79353bf3c24bba851f80bb906fa6bbc27e20bc055fe1bb"
    "66c7bababe4b80b90f2a583a5cd20cbb6d2c5b3b2850c5bb52f3383c9114e1bbfa5d423cfff88fbb8f5b5e3b0ccb38bb"
)
K = np.frombuffer(bytes.fromhex(_KHEX), dtype='<f4').reshape((4, 4, 9, 9))
PHASE = ((1, 1), (0, 1), (0, 0), (1, 0))


def reconstruct(raw):
    R0 = np.asarray(raw, dtype=np.float32)
    h0, w0 = R0.shape
    h, w = h0 + (h0 & 1), w0 + (w0 & 1)
    if (h, w) != (h0, w0):
        R = np.empty((h, w), np.float32)
        R[:h0, :w0] = R0
        R[h0:, :w0] = R0[-1:, :w0]
        R[:h, w0:] = R[:h, w0 - 1:w0]
    else:
        R = R0
    k = K.shape[-1]
    r = k // 2
    n = k * k
    P = np.pad(R, ((r, r), (r, r)), mode='edge')
    F = sliding_window_view(P, (k, k))
    out = np.empty((4, h, w), np.float32)
    for qy in (0, 1):
        for qx in (0, 1):
            q = 2 * qy + qx
            W = K[:, q].reshape(4, n).T.copy()
            Y = F[qy::2, qx::2].reshape(-1, n) @ W
            for c in range(4):
                out[c, qy::2, qx::2] = Y[:, c].reshape(h // 2, w // 2)
    for c, (ry, cx) in enumerate(PHASE):
        out[c, ry::2, cx::2] = R[ry::2, cx::2]
    np.nan_to_num(out, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(out, 0, 255, out=out)
    return np.ascontiguousarray(out[:, :h0, :w0])
