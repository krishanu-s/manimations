import { createRoot } from "react-dom/client";
import GasSimulation from "./GasSimulation";

const root = document.getElementById("gas-simulation-root");
if (root) createRoot(root).render(<GasSimulation />);
