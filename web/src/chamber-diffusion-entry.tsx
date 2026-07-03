import { createRoot } from "react-dom/client";
import ChamberDiffusion from "./ChamberDiffusion";

const root = document.getElementById("chamber-diffusion-root");
if (root) createRoot(root).render(<ChamberDiffusion />);
