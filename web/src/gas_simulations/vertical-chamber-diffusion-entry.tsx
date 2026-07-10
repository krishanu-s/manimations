import { createRoot } from "react-dom/client";
import VerticalChamberDiffusion from "./VerticalChamberDiffusion";

const root = document.getElementById("vertical-chamber-diffusion-root");
if (root) createRoot(root).render(<VerticalChamberDiffusion />);
