from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import json
import random
from typing import Any, Text, Dict, List
from swiplserver import PrologMQI
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import warnings
from rasa_sdk.events import SlotSet
from rasa_sdk.events import ActiveLoop
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz

ruta_archivo_json = "D:\RASA_DATOS\juegos.json"
with open(ruta_archivo_json, "r") as archivo:
    juegos_json = json.load(archivo)

class ActionEnviarJuego(Action):
    def name(self):
        return "action_enviar_juego"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:

        juegos_filtrados = []
        for juego in juegos_json:
            for plataforma, calificacion in juego.get("calificaciones_resenias", {}).items():
                puntuacion = calificacion.get("puntuacion")
                if puntuacion and ("/" in puntuacion):
                    puntuacion_numerica = float(puntuacion.split("/")[0])
                    if puntuacion_numerica > 70 or puntuacion_numerica > 7:
                        juegos_filtrados.append(juego)
                        break 
        if juegos_filtrados:
            juego_aleatorio = random.choice(juegos_filtrados)
            mensaje = f"Te recomiendo el juego {juego_aleatorio['nombre']}, {juego_aleatorio['descripcion']}, la crítica le ha dado un {juego_aleatorio['calificaciones_resenias']['Metacritic']['puntuacion']}, y está a solo {juego_aleatorio['precio']}."
            nombre = juego_aleatorio['nombre']
            dispatcher.utter_message(text=mensaje)
            return [SlotSet("juego", nombre)]
        else:
            dispatcher.utter_message(text="No se encontraron juegos que cumplan con los criterios de puntuación.")

        return []
    
class ActionObtenerJuegoPorGenero(Action):

    def name(self) -> Text:
        return "action_obtener_juegos_por_genero"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        genero = tracker.latest_message.get('text')
        juegos = obtener_juegos_por_genero(genero)
        if juegos is not None:
            if juegos:
                juego_aleatorio = random.choice(juegos)
                mensaje_respuesta = f"Aquí tienes un juego de {genero}: {juego_aleatorio}"
                nombre = str(juego_aleatorio)
            else:
                mensaje_respuesta = f"No encontré juegos de {genero}."
        else:
            mensaje_respuesta = f"Lo siento, hubo un problema al obtener juegos de {genero}."
        if mensaje_respuesta is not None:
            dispatcher.utter_message(text=mensaje_respuesta)
            return [SlotSet("juego", nombre)]
        return []
   
def obtener_juegos_por_genero(genero):
    with PrologMQI(port=8000) as mqi:
        with mqi.create_thread() as prolog_thread:
            prolog_thread.query("consult('C:/Users/Lalo/Documents/Prolog/datos_videojuegos.pl')")
            consulta = f"obtener_juegos_por_genero('{genero}', Juegos)."
            respuesta = list(prolog_thread.query(consulta))
            if respuesta:
                juegos = [registro['Juegos'] for registro in respuesta]
                return juegos
            else:
                return None
            
def modelar_y_visualizar_arbol(csv_file,cantidad):
    try:
     
        df = pd.read_csv(csv_file)
        df = df.drop(columns="nombre")
        df = df.drop (columns="codigo")
        df = df.drop(columns="descripcion")
        df = df.drop(columns="gusto_critica")
        df = df.drop(columns="edad")
        df = df.drop(columns="link")
        df = df.drop(columns="xbox")
        df = df.drop(columns="pc")
        df = df.drop(columns="ps")
        df = df.drop(columns="nintendo")
        df = pd.get_dummies(data=df, drop_first=False)  
        x = df.drop(columns=df.columns[0])
        y = df[df.columns[0]]
        cantidad = cantidad + 1
        print("Cantidad de filas tomadas:")
        print(cantidad)
        x = x.head(cantidad)
        y = y.head(cantidad)      
        model = DecisionTreeClassifier(max_depth=15)   
        model.fit(x, y)       
        dot_data = tree.export_graphviz(model, out_file=None,
            feature_names=x.columns.tolist(),
            class_names=df[df.columns[0]].astype(str).unique().tolist(),
            filled=True, rounded=True,
            special_characters=True)
        graph = graphviz.Source(dot_data)        
        graph.render("arbolPreview")
        global dfAux
        dfAux = df
        return model  

    except Exception as e:
        print(f"Error al modelar y visualizar el árbol de decisión: {e}")

class ActionUsarModelo(Action):
    def name(self):
        return "action_usar_modelo"

    def run(self, dispatcher, tracker, domain):
        try:
            gusto_csv = 'D:/RASA_DATOS/completa.csv'
            cantidad = tracker.get_slot("contador")
            modelo_arbol = modelar_y_visualizar_arbol(gusto_csv,cantidad)
            warnings.filterwarnings("ignore")
            games = pd.read_csv(gusto_csv)
            maximo = len(dfAux)-1
            prediccion_hecha = False
            while prediccion_hecha == False:
                numero_aleatorio = random.randint(0, maximo)
                fila_seleccionada = dfAux.drop(columns=dfAux.columns[0]).iloc[numero_aleatorio]
                fila_seleccionada = fila_seleccionada.values.reshape(1, -1)
                resultado_prediccion = modelo_arbol.predict(fila_seleccionada)
                nombre = games.loc[numero_aleatorio, 'nombre']
                descripcion = games.loc[numero_aleatorio, 'descripcion']
                edad_minima_juego = games.loc[numero_aleatorio,'edad']
                consoleXbox = games.loc[numero_aleatorio,'xbox']
                consolePc = games.loc[numero_aleatorio,'pc']
                consolePs = games.loc[numero_aleatorio,'ps']
                consoleNs = games.loc[numero_aleatorio,'nintendo']
                if consoleXbox == True:
                    console = 'xbox'
                elif consolePc == True:
                    console = 'pc'
                elif consolePs == True:
                    console = 'ps'
                elif consoleNs == True:
                    console = 'nintendo'
                slotConsola = tracker.get_slot("consola")
                if resultado_prediccion == 1:
                    edad = tracker.get_slot("edad") or 18
                    print(f"Consola del usuario: {slotConsola} y edad del juego:{console}")
                    print(f"Edad del usuario: {edad} y edad del juego:{edad_minima_juego}")
                    edad = int(edad)
                    if edad >= edad_minima_juego:
                        if console == slotConsola:
                            dispatcher.utter_message(f"Aqui tienes un juego:" + nombre)
                            dispatcher.utter_message(f"Y un poco acerca del mismo: " + descripcion)
                            prediccion_hecha = True
                            return [SlotSet("juego", nombre)]
        except Exception as e:
            print(f"Error en la acción de recomendación de juegos: {e}")
            
        return []

class ActionTeGustoJuego(Action):
    def name(self):
        return "action_preguntar_juego"
    
    def run(self, dispatcher, tracker, domain):
        try:
            csv_file = 'D:/RASA_DATOS/completa.csv'
            df = pd.read_csv(csv_file)
            if tracker.get_slot("contador") != None:
                if tracker.get_slot("contador") >= 12:
                    dispatcher.utter_message(f"Ya tenemos datos suficientes. ")
                    action_usar_modelo = ActionUsarModelo()
                    return action_usar_modelo.run(dispatcher, tracker, domain)
                else:          
                    contador = tracker.get_slot("contador") or 0
                    posicion_deseada = contador
                    iterador_filas = df.iterrows()
                    for _ in range(posicion_deseada):
                        try:
                            index, row = next(iterador_filas)
                        except StopIteration:                 
                            break
                    index, row = next(iterador_filas)
                    nombre_videojuego = row['nombre']
                    descripcion = row['descripcion']
                    dispatcher.utter_message(f"{nombre_videojuego}")
                    dispatcher.utter_message( " Descripcion: "+ descripcion)
                    contador = contador
                    return [SlotSet("contador", contador)]
            else:    
                contador = tracker.get_slot("contador") or 0
                posicion_deseada = contador
                iterador_filas = df.iterrows()     
                for _ in range(posicion_deseada):
                    try:
                        index, row = next(iterador_filas)
                    except StopIteration:                           
                        break            
                index, row = next(iterador_filas)     
                nombre_videojuego = row['nombre']
                descripcion = row['descripcion']
                dispatcher.utter_message(f"¿Te gustó el videojuego: {nombre_videojuego}?"+ " Por si no lo conoces aqui una breve descripcion: "+ descripcion)
                contador = contador
                return [SlotSet("contador", contador)]

        except Exception as e:
            print(f"Error al modificar el archivo CSV: {e}")

class CambiarGustoJuego(Action):
    def name(self):
        return "action_cambiar_gusto_juego"
    
    def run(self, dispatcher, tracker, domain):
        try:
            csv_file = 'D:/RASA_DATOS/completa.csv'
            df = pd.read_csv(csv_file)
            contador = tracker.get_slot("contador") or 0
            posicion_deseada = contador
            print(contador)
            iterador_filas = df.iterrows()        
            for _ in range(posicion_deseada):
                try:
                    index, row = next(iterador_filas)
                except StopIteration:                 
                    break
            index, row = next(iterador_filas)  
            ultimo_intent = tracker.latest_message['intent']['name']
            if ultimo_intent == "affirm":
                gusto_usuario = True
                contador += 1
            elif ultimo_intent == "deny":
                gusto_usuario = False
                contador += 1
            else:
                gusto_usuario = False
            df.at[index, 'gusto'] = gusto_usuario        
            df.to_csv(csv_file, index=False)
            return [SlotSet("contador", contador)]
        
        except Exception as e:
            print(f"Error al modificar el archivo CSV: {e}")

class RecomendacionSegunCriticaAction(Action):
    def name(self):
        return "action_recomendacion_segun_critica"
    
    def run(self, dispatcher, tracker, domain):
        try:    
            csv_file = 'D:/RASA_DATOS/completa.csv'
            df = pd.read_csv(csv_file)
            df2 = df 
            df = df.drop (columns="gusto")
            df = df.drop (columns="codigo")
            df = df.drop(columns="descripcion")
            df = df.drop(columns="nombre")
            df = df.drop(columns="edad")
            df = df.drop(columns="link")
            df = df.drop(columns="xbox")
            df = df.drop(columns="pc")
            df = df.drop(columns="ps")
            df = df.drop(columns="nintendo")
            label_encoder = LabelEncoder()
            df['empresa'] = label_encoder.fit_transform(df['empresa'])
            df = df.astype(int)
            X = df[['accion','aventura','rpg','sandbox','puzzle','fantasia','realismo','terror','ciencia ficcion','vr','violencia','lucha','battle royale','shooter','roguelike','metroidvania','mmo','estrategia','aventura grafica','aventura conversacional y narrativa','plataforma','musical','carreras','simulacion','supervivencia','graficos realistas', 'online' , 'empresa', 'anio', 'puntuacion metacritic']].values
            y = df['gusto_critica'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,  stratify=y)
            model = keras.Sequential([
                keras.layers.Input(shape=(X_train.shape[1],)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid'),
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f'Loss: {loss}, Accuracy: {accuracy}')
            df = df.drop (columns="gusto_critica")
            prediccion_hecha = False
            iteracion = 0
            while prediccion_hecha == False:
                maximo = len(df)-1
                numero_aleatorio = random.randint(0, maximo)
                sample_game = df.iloc[numero_aleatorio]
                sample_game = sample_game.values.reshape(1, -1)
                nombre_juego = df2.loc[numero_aleatorio,'nombre']
                prediction = model.predict(sample_game)
                print(f"========={nombre_juego}========")
                if prediction >= 0.5 or iteracion >= 50:
                    dispatcher.utter_message(f"El juego {nombre_juego} puede cumplir con lo que buscas")
                    prediccion_hecha = True
                iteracion = iteracion + 1
            return [SlotSet("juego", nombre_juego)]

        except Exception as e:
            print(f"Error en main2: {e}")

class EdadAction(Action):
    def name(self):
        return "action_edad"
    
    def run(self, dispatcher, tracker, domain):
        try:
            mensaje = next(tracker.get_latest_entity_values("edad"), None)
            edad = mensaje  
            print(edad)
            return [SlotSet("edad", edad)]
        
        except Exception as e:
            print(f"Error en class EdadAction(Action): {e}")

class UsernameAction(Action):
    def name(self):
        return "action_username"
    
    def run(self, dispatcher, tracker, domain):
        try:
            mensaje = next(tracker.get_latest_entity_values("username"), None)
            nombre = mensaje  
            print(nombre)
            if nombre == None:
                return [ActiveLoop("name_form")]
            return [SlotSet("username", nombre)]
        
        except Exception as e:
            print(f"Error en class UsernameAction(Action): {e}")

class MostrarLinkJuegoAction(Action):
    def name(self):
        return "mostrar_link_juego"
    
    def run(self, dispatcher, tracker, domain):
        try:
            csv_file = 'D:/RASA_DATOS/completa.csv'
            df = pd.read_csv(csv_file)
            game = tracker.get_slot("juego")
            fila = df.loc[df['nombre'] == game]
            if len(fila) != 0:
                link=fila['link'].values[0]
                dispatcher.utter_message(f"Aqui tienes un link de compra {link}")
            else:
                dispatcher.utter_message("No pude encontrar el link de compra pero aqui seguro lo encontraras")
                dispatcher.utter_message("https://store.steampowered.com/")
           
        
        except Exception as e:
            print(f"Error en class MostrarLinkJuegoAction(Action): {e}")

class MostrarDescripcionJuegoAction(Action):
    def name(self):
        return "mostrar_descripcion_juego"
    
    def run(self, dispatcher, tracker, domain):
        try:
            csv_file = 'D:/RASA_DATOS/completa.csv'
            df = pd.read_csv(csv_file)
            game = tracker.get_slot("juego")
            fila = df.loc[df['nombre'] == game]
            if len(fila) != 0:
                descripcion=fila['descripcion'].values[0]
                dispatcher.utter_message(f"Aqui un poco acerca del juego: {descripcion}")
            else:
                dispatcher.utter_message("No pude encontrar informacion pero aqui seguro la encontraras")
                dispatcher.utter_message("https://www.metacritic.com/game/")
           
        
        except Exception as e:
            print(f"Error en class MostrarLinkJuegoAction(Action): {e}")

class UsernameAction(Action):
    def name(self):
        return "action_username"
    
    def run(self, dispatcher, tracker, domain):
        try:
            mensaje = next(tracker.get_latest_entity_values("username"), None)
            nombre = mensaje  
            print(nombre)
            if nombre == None:
                return [ActiveLoop("name_form")]
            dispatcher.utter_message(f"Mucho gusto {nombre}")
            return [SlotSet("username", nombre)]
        
        except Exception as e:
            print(f"Error en class UsernameAction(Action): {e}")

class UsernameVaciarAction(Action):
    def name(self):
        return "vaciar_user_name"
    
    def run(self, dispatcher, tracker, domain):
        try:
            return [SlotSet("username", None)]
        
        except Exception as e:
            print(f"Error en class UsernameVaciarAction(Action): {e}")



class GuardarDatosUsuario(Action):
    def name(self):
        return "action_guardar_datos_usuario"

    def run(self, dispatcher, tracker, domain):
        try:
            nombre_usuario = tracker.get_slot("username")
            edad_usuario = tracker.get_slot("edad")
            csv_file = 'D:/RASA_DATOS/usuarios.csv'
            df = pd.read_csv(csv_file)
            registro = df[df["nombre"] == nombre_usuario]
            if registro.empty:
                nuevo_df = pd.DataFrame({"nombre": [nombre_usuario], "edad": [edad_usuario]})
                df = pd.concat([nuevo_df, df], ignore_index=True)
                df.to_csv(csv_file, index=False)
                dispatcher.utter_message(f"¡Gracias, {nombre_usuario}! Tus datos han sido guardados.")
            else:
                dispatcher.utter_message(f"El usuario {nombre_usuario} ya fue utilizado")

        except Exception as e:
            print(f"Error en action_guardar_datos_usuario: {e}")

        return []

class CargarDatosUsuario(Action):
    def name(self):
        return "action_cargar_datos_usuarios"

    def run(self, dispatcher, tracker, domain):
        try:
            csv_file = 'D:/RASA_DATOS/usuarios.csv'
            username = tracker.get_slot("username")
            df = pd.read_csv(csv_file)
            registro = df[df["nombre"] == username]
            if not registro.empty:
                nombre_usuario = registro.iloc[0]["nombre"]
                edad_usuario = str(registro.iloc[0]["edad"])  
                dispatcher.utter_message(f"{nombre_usuario}, tus datos de usario han sido cargados correctamente, recuerdo que tenias {edad_usuario} anios")
                return [SlotSet("username", nombre_usuario), SlotSet("edad", edad_usuario)]
            else:
                dispatcher.utter_message(f"No hay datos para el username ingresado")

        except Exception as e:
            print(f"Error en action_cargar_datos_usuario: {e}")

        return []

class VerDatosAction(Action):
    def name(self):
        return "action_ver_datos"

    def run(self, dispatcher, tracker, domain):
        try:            
            username = tracker.get_slot("username")
            edad = tracker.get_slot("edad")
            if username != None:
                dispatcher.utter_message(f"Tu nombre de usuario es {username}")
            if edad != None:
                dispatcher.utter_message(f"Tienes {edad} anios")

        except Exception as e:
            print(f"Error en VerDatosAction: {e}")

        return []

class GuardarConsolaAction(Action):
    def name(self):
        return "action_save_console"

    def run(self, dispatcher, tracker, domain):
        try:
            ultimo_intent = tracker.latest_message['intent']['name']
            if ultimo_intent == "xbox_console":
                console = "xbox"
            elif ultimo_intent == "pc_console":
                console = "pc"
            elif ultimo_intent == "playstation_console":
                console = "ps"
            elif ultimo_intent == "nintendo_console":
                console = "ns"
            else:
                console = "pc"
            return [SlotSet("consola", console)]
        
        except Exception as e:
            print(f"Error en VerDatosAction: {e}")

        return []

class GoodbyeAction(Action):
    def name(self):
        return "action_goodbye"

    def run(self, dispatcher, tracker, domain):
        try:
            nombre = tracker.get_slot("username")
            if nombre == None:
                dispatcher.utter_template("utter_goodbye", tracker)
            else:
                dispatcher.utter_template("utter_goodbye_with_name", tracker)
        
        except Exception as e:
            print(f"Error en GoodbyeAction: {e}")

        return []

class BuscarJuegoAction(Action):
    def name(self):
        return "action_search_game"

    def run(self, dispatcher, tracker, domain):
        try:
            mensaje = tracker.get_slot("juego")
            print(mensaje)
            print("entra")
            nombre_juego, indice_juego = buscar_juego_similar(mensaje)
            print("sale")
            if nombre_juego != "Juego no encontrado":
                dispatcher.utter_message(f"Y te gusta el '{nombre_juego}'?")
                print(indice_juego)
                return [SlotSet("contador", indice_juego), SlotSet("juego", nombre_juego)]
            else:
                dispatcher.utter_message("Lo siento, no reconozco ese juego.")
        except Exception as e:
            print(f"Error en BuscarJuegoAction: {e}")

        return []

def buscar_juego_similar(mensaje, threshold=80):
    
    df = pd.read_csv('D:/RASA_DATOS/completa.csv')
    best_match = None
    best_ratio = 0
    best_index = -1

    for index, juego in enumerate(df['nombre']):
        ratio = fuzz.token_set_ratio(mensaje, juego.lower())
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = juego
            best_index = index

    if best_ratio >= threshold:
        return best_match, best_index
    else:
        return "Juego no encontrado", -1
