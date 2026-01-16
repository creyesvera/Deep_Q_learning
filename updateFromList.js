let episodeStates = [];

console.log("animate.js cargado");


const params = new URLSearchParams(window.location.search);
const episode = params.get("ep") || "episode_01_states.json";

// cargar datos
fetch(`episodes/${episode}`)
  .then(r => r.json())
  .then(data => {
    episodeStates = data;
    requestAnimationFrame(step);
  });






function mapRange(value, inMin, inMax, outMin, outMax) {
  return (
    (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin
  );
}

const TEMP_MIN = 23; // °C (ajústalo si lo sabes exacto)
const TEMP_MAX = 33; // °C


const tempIN = document.getElementById("Variable");
const tempOUT = document.getElementById("Variable_2");

const encendido = document.getElementById("Encendido");
const apagado = document.getElementById("Apagado");

const bajar_temp = document.getElementById("Bajar temperatura");
const subir_temp = document.getElementById("Subir temperatura");

const laboratorio_abierto = document.getElementById("Abierto");
const clases_practicas = document.getElementById("Clases");
const laboratorio_cerrado = document.getElementById("Cerrado");

const frio = document.getElementById("Frio");
const neutro = document.getElementById("Neutro");
const calor = document.getElementById("Calor");

const icons_people = {
  centro: document.getElementById("Centro"),
  iP: document.getElementById("i persona p"),
  dP: document.getElementById("d persona p"),
  iG: document.getElementById("I persona G"),
  dG: document.getElementById("D persona G")
};

const text_ubicacion = document.getElementById("Ubicacion Relativa Variable");



const minHeight = 124;
const maxHeight = 315

const clasesMap = {
  0: laboratorio_abierto,
  1: clases_practicas,
  2: laboratorio_cerrado
};

const thermalMap = {
  0: frio,
  1: neutro,
  2: calor
};

const actionMap = {
  0: encendido, //Prender ventilación
  1: bajar_temp,      //Bajar temperatura
  //2: no_hacer_nada,    //No hacer nada
  3: subir_temp,      //Subir temperatura
  4: apagado       //Apagar ventilación
};

const ubicacionTextMap = {
  0: "Dispersos",
  1: "Agrupadas cerca de ventilación",
  2: "Agrupadas lejos"
};





function peopleLevel(n) {
  if (n === 0) return "none";
  if (n <= 5) return "low";
  if (n <= 12) return "medium";
  return "high";
}

function hideAllPeople() {
  Object.values(icons_people).forEach(el => {
    el.style.display = "none";
  });
}

function renderPeopleCount(n) {
  hideAllPeople();

  const level = peopleLevel(n);

  if (level === "none") return;

  if (level === "low") {
    icons_people.centro.style.display = "inline";
  }

  if (level === "medium") {
    icons_people.centro.style.display = "inline";
    icons_people.iG.style.display = "inline";
    icons_people.dG.style.display = "inline";
  }

  if (level === "high") {
    Object.values(icons_people).forEach(el => {
      el.style.display = "inline";
    });
  }
}






let i = 0;

function updateFromList() {

  // Actualizar barras de temperatura
    // Temperatura interior
    const h_tempIn = mapRange(
      episodeStates[i].temp_in,
      TEMP_MIN,
      TEMP_MAX,
      minHeight,
      maxHeight
    );
    tempIN.setAttribute("height", h_tempIn);

    // Temperatura exterior
    const h_tempOut = mapRange(
      episodeStates[i].temp_out,
      TEMP_MIN,
      TEMP_MAX,
      minHeight,
      maxHeight
    );
    tempOUT.setAttribute("height", h_tempOut);

    // Actualizar estado del aire acondicionado
    if (episodeStates[i].ac) {
        encendido.style.display = "inline";
        apagado.style.display = "none";
    } else {
        encendido.style.display = "none";
        apagado.style.display = "inline";

        bajar_temp.style.display = "none";
        subir_temp.style.display = "none";
    }


    //Actualizar clases
    
    laboratorio_abierto.style.display = "none";
    clases_practicas.style.display = "none";
    laboratorio_cerrado.style.display = "none";

    clasesMap[episodeStates[i].clases].style.display = "inline";


    //Actualizar opinion termica
    
    frio.style.display = "none";
    neutro.style.display = "none";
    calor.style.display = "none";

    thermalMap[episodeStates[i].thermal].style.display = "inline";


    
    //Actualizar accion tomada
    if (episodeStates[i].action !== 2) {
        bajar_temp.style.display = "none";
        subir_temp.style.display = "none";
        actionMap[episodeStates[i].action].style.display = "inline"; 
    }
    
    


    //Actualizar numero de personas
    renderPeopleCount(episodeStates[i].n_personas);

    //Actualizar ubicacion de las personas
    text_ubicacion.textContent = ubicacionTextMap[episodeStates[i].ubicacion];

    //text_ubicacion.textContent =    ubicacionTextMap[state.ubicacion] ?? "—";


    //el.style.opacity = 0;
    //el.style.display = "inline";
    //requestAnimationFrame(() => el.style.opacity = 1);

    i = (i + 1) % episodeStates.length;
}

// velocidad de animación (ms)
setInterval(updateFromList, 150);