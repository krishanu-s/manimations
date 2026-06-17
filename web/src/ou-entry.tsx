import { createRoot } from "react-dom/client";
import OrnsteinUhlenbeck from "./OrnsteinUhlenbeck";

const root = document.getElementById("ou-root");
if (root) createRoot(root).render(<OrnsteinUhlenbeck />);
