# python utils/test/fig.py

from pyfiglet import Figlet

# 귀여운 글꼴 리스트
cute_fonts = [
    'devilish', 'diamond', 'diet_cola', 'digital', 'doh', 'doom',
    'eftitalic', 'eftiwall', 'eftiwater', 'efti_robot', 'electronic', 'elite', 'epic', 'etcrvs__', 'e__fist_', 'f15_____', 'faces_of'
]

select_fonts = ['soft','ansi_shadow','ansi_shadow','blocky','bloody','big','calvin_s','chunky','elite','electronic','doom',

                ]
text = "DevBear"

for font in cute_fonts:
    try:
        f = Figlet(font=font)
        print(f"\n[ {font} ]\n")
        print(f.renderText(text))
    except Exception as e:
        print(f"[ERROR with font {font}]: {e}")


"""

[  'cosmic', 'cosmike', '
cour', 'courb', 'courbi', 'couri', 'crawford', 'crawford2', 'crazy', 'cricket', 'cursive', 'cyberlarge', 'cybermedium', 'cybersmall', 'cygnet', 'c_ascii_', 'c_consen', 'danc4', 'dancing_font',
 ,  'dos_rebel', 'dotmatrix', 'double', 'double_blocky', 'double_shorts', 'drpepper', 'druid___', 'dwhistled', 'd_dragon', 'ebbs_1__', 'ebbs_2__', 'eca_____', 'eftichess', 'eftifont', 'eftipiti', 'efti
robot', , 'fairligh', 'fair_mea', 'fantasy_', 'fbr12___', 'fbr1
____', 'fbr2____', 'fbr_stri', 'fbr_tilt', 'fender', 'filter', 'finalass', 'fireing_', 'fire_font-k', 'fire_font-s', 'flipped', 'flower_power', 'flyn_sh', 'fourtops', 'fp1_____', 'fp2_____', '
fraktur', , 'future_7', 'future_8', 'fuzzy', 'gauntlet', 'georgi16', 'georgia11', 'gh
ost', 'ghost_bo', 'ghoulish', 'glenyn', 'goofy', 'gothic', 'gothic__', 'graceful', 'gradient', 'graffiti', 'grand_pr', 'greek', 'green_be', 'hades___', 'heart_left', 'heart_right', 'heavy_me',
 'helv', 'helvb', 'helvbi', 'helvi', 'henry_3d', 'heroboti', 'hex', 'hieroglyphs', 'high_noo', 'hills___', 'hollywood', 'home_pak', 'horizontal_left', 'horizontal_right', 'house_of', 'hypa_bal
', 'hyper___', 'icl-1900', 'impossible', 'inc_raw_', 'invita', 'isometric1', 'isometric2', 'isometric3', 'isometric4', 'italic', 'italics_', 'ivrit', 'jacky', 'jazmine', 'jerusalem', 'joust___
', 'js_block_letters', 'js_bracket_letters', 'js_capital_curves', 'js_cursive', 'js_stick_letters', 'katakana', 'kban', 'keyboard', 'kgames_i', 'kik_star', 'knob', 'konto', 'konto_slant', 'kra
k_out', 'larry3d', 'lazy_jon', 'lcd', 'lean', 'letters', 'letterw3', 'letter_w', 'lexible_', 'lil_devil', 'line_blocks', 'linux', 'lockergnome', 'madrid', 'mad_nurs', 'magic_ma', 'marquee', 'm
aster_o', 'maxfour', 'mayhem_d', 'mcg_____', 'merlin1', 'merlin2', 'mig_ally', 'mike', 'mini', 'mirror', 'mnemonic', 'modern__', 'modular', 'morse', 'morse2', 'moscow', 'mshebrew210', 'muzzle'
, 'nancyj-fancy', 'nancyj-improved', 'nancyj-underlined', 'nancyj', 'new_asci', 'nfi1____', 'nipples', 'notie_ca', 'npn_____', 'nscript', 'ntgreek', 'nvscript', 'o8', 'octal', 'odel_lak', 'ogr
e', 'ok_beer_', 'old_banner', 'os2', 'outrun__', 'pacos_pe', 'panther_', "patorjk's_cheese", 'patorjk-hex', 'pawn_ins', 'pawp', 'peaks', 'pebbles', 'pepper', 'phonix__', 'platoon2', 'platoon_'
, 'pod_____', 'poison', 'puffy', 'puzzle', 'pyramid', 'p_skateb', 'p_s_h_m_', 'r2-d2___', 'radical_', 'rad_phan', 'rad_____', 'rainbow_', 'rally_s2', 'rally_sp', 'rammstein', 'rampage_', 'rast
an__', 'raw_recu', 'rci_____', 'rectangles', 'red_phoenix', 'relief', 'relief2', 'rev', 'ripper!_', 'road_rai', 'rockbox_', 'rok_____', 'roman', 'roman___', 'rot13', 'rotated', 'rounded', 'row
ancap', 'rozzo', 'runic', 'runyc', 'sans', 'sansb', 'sansbi', 'sansi', 'santa_clara', 'sblood', 'sbook', 'sbookb', 'sbookbi', 'sbooki', 'script', 'script__', 'serifcap', 'shadow', 'shimrod', '
short', 'skateord', 'skateroc', 'skate_ro', 'sketch_s', 'slant', 'slant_relief', 'slide', 'slscript', 'sl_script', 'small', 'small_caps', 'small_poison', 'small_shadow', 'small_slant', 'smisom
e1', 'smkeyboard', 'smscript', 'smshadow', 'smslant', 'smtengwar', 'sm______', 'soft', 'space_op', 'spc_demo', 'speed', 'spliff', 'stacey', 'stampate', 'stampatello', 'standard', 'starwars', '
star_strips', 'star_war', 'stealth_', 'stellar', 'stencil1', 'stencil2', 'stforek', 'stick_letters', 'stop', 'straight', 'street_s', 'stronger_than_all', 'sub-zero', 'subteran', 'super_te', 's
wamp_land', 'swan', 'sweet', 'tanja', 'tav1____', 'taxi____', 'tec1____', 'tecrvs__', 'tec_7000', 'tengwar', 'term', 'test1', 'the_edge', 'thick', 'thin', 'this', 'thorned', 'threepoint', 'tic
ks', 'ticksslant', 'tiles', 'times', 'timesofl', 'tinker-toy', 'ti_pan__', 'tomahawk', 'tombstone', 'top_duck', 'train', 'trashman', 'trek', 'triad_st', 'ts1_____', 'tsalagi', 'tsm_____', 'tsn
_base', 'tty', 'ttyb', 'tubular', 'twin_cob', 'twisted', 'twopoint', 'type_set', 't__of_ap', 'ucf_fan_', 'ugalympi', 'unarmed_', 'univers', 'usaflag', 'usa_pq__', 'usa_____', 'utopia', 'utopia
b', 'utopiabi', 'utopiai', 'varsity', 'vortron_', 'war_of_w', 'wavy', 'weird', 'wet_letter', 'whimsy', 'wow', 'xbrite', 'xbriteb', 'xbritebi', 'xbritei', 'xchartr', 'xchartri', 'xcour', 'xcour
b', 'xcourbi', 'xcouri', 'xhelv', 'xhelvb', 'xhelvbi', 'xhelvi', 'xsans', 'xsansb', 'xsansbi', 'xsansi', 'xsbook', 'xsbookb', 'xsbookbi', 'xsbooki', 'xtimes', 'xtty', 'xttyb', 'yie-ar__', 'yie_ar_k', 'z-pilot_', 'zig_zag_', 'zone7___']



"""