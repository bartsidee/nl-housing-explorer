# Provincie mapping voor Nederlandse gemeenten (2023)
# Bron: CBS / PDOK Gebiedsindelingen 2023

PROVINCIE_GEMEENTE_MAPPING = {
    'Groningen': [
        'Groningen', 'Oldambt', 'Stadskanaal', 'Veendam', 'Pekela', 'Westerwolde',
        'Midden-Groningen', 'Westerkwartier', 'Het Hogeland', 'Eemsdelta'
    ],
    'Friesland': [
        'Leeuwarden', 'Smallingerland', 'Heerenveen', 'Súdwest-Fryslân', 'De Fryske Marren',
        'Waadhoeke', 'Noardeast-Fryslân', 'Harlingen', 'Ooststellingwerf', 'Weststellingwerf',
        'Achtkarspelen', 'Tytsjerksteradiel', 'Opsterland', 'Dantumadiel', 'Ameland',
        'Het Bildt', 'Terschelling', 'Vlieland', 'Schiermonnikoog'
    ],
    'Drenthe': [
        'Assen', 'Emmen', 'Coevorden', 'Hoogeveen', 'Aa en Hunze', 'Borger-Odoorn',
        'Midden-Drenthe', 'Noordenveld', 'Tynaarlo', 'Westerveld', 'De Wolden'
        , 'Meppel'
    ],
    'Overijssel': [
        'Zwolle', 'Enschede', 'Almelo', 'Deventer', 'Hengelo', 'Hengelo (O.)', 'Oldenzaal', 'Kampen',
        'Hardenberg', 'Raalte', 'Tubbergen', 'Borne', 'Dalfsen', 'Dinkelland',
        'Haaksbergen', 'Hellendoorn', 'Losser', 'Olst-Wijhe', 'Rijssen-Holten',
        'Staphorst', 'Twenterand', 'Wierden', 'Zwartewaterland', 'Steenwijkerland'
        , 'Hof van Twente', 'Ommen'
    ],
    'Flevoland': [
        'Almere', 'Lelystad', 'Dronten', 'Noordoostpolder', 'Urk', 'Zeewolde'
    ],
    'Gelderland': [
        'Arnhem', 'Nijmegen', 'Apeldoorn', 'Ede', 'Doetinchem', 'Zutphen', 'Harderwijk',
        'Tiel', 'Wageningen', 'Zevenaar', 'Aalten', 'Barneveld', 'Berg en Dal',
        'Berkelland', 'Beuningen', 'Bronckhorst', 'Brummen', 'Buren', 'Culemborg',
        'Druten', 'Duiven', 'Elburg', 'Epe', 'Ermelo', 'Hattem', 'Heerde',
        'Heumen', 'Lingewaard', 'Lochem', 'Maasdriel', 'Montferland', 'Neder-Betuwe',
        'Nijkerk', 'Nunspeet', 'Oldebroek', 'Oost Gelre', 'Oude IJsselstreek',
        'Overbetuwe', 'Putten', 'Renkum', 'Rheden', 'Rozendaal', 'Scherpenzeel',
        'Voorst', 'Wageningen', 'West Betuwe', 'West Maas en Waal', 'Winterswijk',
        'Wijchen', 'Zutphen', 'Doesburg', 'Westervoort', 'Zaltbommel'
    ],
    'Utrecht': [
        'Utrecht', 'Amersfoort', 'Nieuwegein', 'Veenendaal', 'Houten', 'Zeist',
        'Woerden', 'IJsselstein', 'Bunnik', 'De Bilt', 'Baarn', 'Bunschoten',
        'Eemnes', 'Leusden', 'Lopik', 'Montfoort', 'Oudewater', 'Renswoude',
        'Rhenen', 'Soest', 'Stichtse Vecht', 'Utrecht', 'Utrechtse Heuvelrug',
        'Vijfheerenlanden', 'Wijk bij Duurstede', 'Woudenberg'
        , 'De Ronde Venen'
    ],
    'Noord-Holland': [
        'Amsterdam', 'Haarlem', 'Zaanstad', 'Haarlemmermeer', 'Alkmaar', 'Amstelveen',
        'Purmerend', 'Hoorn', 'Velsen', 'Den Helder', 'Hilversum', 'Heerhugowaard',
        'Beverwijk', 'Uithoorn', 'Landsmeer', 'Wormerland', 'Waterland', 'Oostzaan',
        'Beemster', 'Bergen (NH.)', 'Blaricum', 'Bloemendaal', 'Castricum', 'Diemen',
        'Edam-Volendam', 'Enkhuizen', 'Gooise Meren', 'Heemskerk', 'Heemstede',
        'Hollands Kroon', 'Huizen', 'Koggenland', 'Langedijk', 'Laren', 'Laren (NH.)', 'Medemblik',
        'Oostzaan', 'Opmeer', 'Ouder-Amstel', 'Schagen', 'Stede Broec', 'Texel',
        'Uitgeest', 'Velsen', 'Weesp', 'Zandvoort', 'Drechterland', 'Edam-Volendam',
        'Aalsmeer', 'Dijk en Waard', 'Heiloo', 'Wijdemeren'
    ],
    'Zuid-Holland': [
        'Rotterdam', "'s-Gravenhage", 'Den Haag', 'Zoetermeer', 'Leiden', 'Dordrecht',
        'Alphen aan den Rijn', 'Westland', 'Delft', 'Capelle aan den IJssel', 'Spijkenisse',
        'Schiedam', 'Vlaardingen', 'Gouda', 'Katwijk', 'Ridderkerk', 'Papendrecht',
        'Hellevoetsluis', 'Rijswijk', 'Rijswijk (ZH.)', 'Leidschendam-Voorburg', 'Pijnacker-Nootdorp',
        'Nissewaard', 'Krimpen aan den IJssel', 'Waddinxveen', 'Wassenaar', 'Voorschoten',
        'Albrandswaard', 'Alblasserdam', 'Barendrecht', 'Bodegraven-Reeuwijk', 'Brielle',
        'Goeree-Overflakkee', 'Gorinchem', 'Hardinxveld-Giessendam', 'Hendrik-Ido-Ambacht',
        'Hillegom', 'Hoeksche Waard', 'Kaag en Braassem', 'Krimpenerwaard', 'Lansingerland',
        'Lisse', 'Maassluis', 'Molenlanden', 'Nieuwkoop', 'Noordwijk', 'Oegstgeest',
        'Papendrecht', 'Sliedrecht', 'Westvoorne', 'Zwijndrecht', 'Zuidplas',
        'Voorne aan Zee', 'Midden-Delfland', 'Leiderdorp', 'Teylingen', 'Zoeterwoude'
    ],
    'Zeeland': [
        'Middelburg', 'Middelburg (Z.)', 'Vlissingen', 'Terneuzen', 'Goes', 'Hulst', 'Borsele',
        'Kapelle', 'Noord-Beveland', 'Reimerswaal', 'Schouwen-Duiveland',
        'Sluis', 'Tholen', 'Veere'
    ],
    'Noord-Brabant': [
        'Eindhoven', 'Tilburg', "'s-Hertogenbosch", 'Breda', 'Helmond', 'Oss',
        'Roosendaal', 'Bergen op Zoom', 'Uden', 'Oosterhout', 'Veghel', 'Waalwijk',
        'Etten-Leur', 'Valkenswaard', 'Deurne', 'Boxmeer', 'Boxtel', 'Cuijk',
        'Best', 'Veldhoven', 'Bernheze', 'Cranendonck', 'Geertruidenberg',
        'Gilze en Rijen', 'Goirle', 'Grave', 'Heusden', 'Hilvarenbeek', 'Land van Cuijk',
        'Loon op Zand', 'Meierijstad', 'Moerdijk', 'Nuenen', 'Oirschot',
        'Oisterwijk', 'Rucphen', 'Sint-Michielsgestel', 'Someren', 'Son en Breugel',
        'Steenbergen', 'Waalre', 'Zundert', 'Aalburg', 'Alphen-Chaam', 'Asten',
        'Baarle-Nassau', 'Bergeijk', 'Bladel', 'Boekel', 'Dongen', 'Drimmelen',
        'Eersel', 'Geldrop-Mierlo', 'Gemert-Bakel', 'Halderberge', 'Haaren',
        'Laarbeek', 'Landerd', 'Mill en Sint Hubert', 'Oirschot', 'Reusel-De Mierden',
        'Schijndel', 'Sint Anthonis', 'Werkendam', 'Woensdrecht'
        , 'Altena', 'Heeze-Leende', 'Maashorst', 'Nuenen, Gerwen en Nederwetten', 'Vught'
    ],
    'Limburg': [
        'Maastricht', 'Venlo', 'Sittard-Geleen', 'Heerlen', 'Roermond', 'Kerkrade',
        'Weert', 'Venray', 'Brunssum', 'Beek', 'Beek (L.)', 'Valkenburg aan de Geul', 'Stein', 'Stein (L.)',
        'Beekdaelen', 'Beesel', 'Bergen (L.)', 'Echt-Susteren', 'Eijsden-Margraten', 'Gennep', 'Gulpen-Wittem',
        'Horst aan de Maas', 'Landgraaf', 'Leudal', 'Maasgouw', 'Meerssen', 'Mook en Middelaar',
        'Nederweert', 'Peel en Maas', 'Roerdalen', 'Simpelveld', 'Stein', 'Vaals',
        'Valkenburg aan de Geul', 'Venray', 'Voerendaal', 'Weert'
    ]
}

def get_provincie_for_gemeente(gemeente_naam: str) -> str:
    """Get provincie name for a gemeente"""
    for provincie, gemeenten in PROVINCIE_GEMEENTE_MAPPING.items():
        if gemeente_naam in gemeenten:
            return provincie
    return None

def get_all_provincies() -> list:
    """Get list of all provincies"""
    return sorted(PROVINCIE_GEMEENTE_MAPPING.keys())

def get_gemeenten_in_provincie(provincie: str) -> list:
    """Get list of gemeenten in a provincie"""
    return PROVINCIE_GEMEENTE_MAPPING.get(provincie, [])
