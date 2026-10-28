# LPM (Large Physics foundation Model)
## Correlation versus causation

Most AI today, LLMs included, learns statistical associations. A model trained on weather records learns that a particular pressure pattern is usually followed by rain. That is useful for forecasting, but it doesn't tell you whether the pressure pattern causes the rain, whether both come from some third factor, or what would happen if you could change the pressure pattern.

Judea Pearl, the computer scientist who formalized much of causal inference, describes three levels of reasoning:

| Level | Question | Weather example |
|---|---|---|
| 1. Association | What is likely to happen, given what I see? | "Rain is likely tomorrow." |
| 2. Intervention | What happens if I *do* something? | "If we seed these clouds now, does rainfall increase downwind?" |
| 3. Counterfactual | What would have happened if things had been different? | "Would this flood have occurred if the storm had tracked 50 km east?" |

Today's forecasting models, both physics simulations and ML systems, mostly operate at level 1. Causal Labs is aiming for levels 2 and 3: a model that can answer "what if we act?" and not just "what's next?"

**Why physics is a promising place to learn causality**

The company's bet is that physical systems are unusually good training material for causal reasoning:

- **Causes come before effects.** In physical data, time ordering is reliable: a pressure drop precedes the storm. Text and images don't carry that structure as cleanly.
- **The rules don't change.** Conservation of mass, energy and momentum hold everywhere and always. A model that has truly learned these constraints should behave sensibly even in situations it hasn't seen, which is the hallmark of causal understanding rather than memorized patterns.
- **Predictions get checked quickly.** A weather forecast is proven right or wrong within hours or days. That gives fast, objective feedback, unlike a language model, where "correct" is often subjective.
- **Real interventions happen.** Weather is mostly observed, not experimented on, but natural and human perturbations do occur: volcanic eruptions, wildfire smoke, cloud seeding operations. These act as partial natural experiments.

## From prediction to intervention

If a model has really learned how the atmosphere evolves, you can perturb its internal state and simulate the consequences. For example, you could add aerosol particles over a region, cool the ocean surface ahead of a hurricane, or change irrigation over farmland, and watch how the model's future diverges. Their job postings describe exactly this: a planning layer that takes a goal (such as "minimize crop loss" or "reduce storm intensity") and searches for actions that achieve it.

In the near term, the "actions" are ordinary business decisions: when a farmer should harvest or spray, when a grid operator should shift load, when an airline should reroute. "Weather control" is the long-term aspiration.

**Why it could generalize beyond weather**

The underlying hope is similar to how LLMs learned general language skills from vast text. A model trained at scale on physical dynamics might learn reusable concepts, such as fluid flow, diffusion, heat transfer and reaction rates, that transfer to other physical systems. In biology, that could mean predicting how a cell responds if you knock out a gene or add a drug. In materials science, it could mean predicting how a material's properties change if you alter its composition or processing. In both cases the questions are interventional ("what if I change X?"), which is why the causal framing matters more than plain prediction.

**The hard parts**

There are good reasons for skepticism:

- **Accurate prediction doesn't prove causal understanding.** A model can forecast well by exploiting correlations that break down when you intervene. Testing interventional claims on the real atmosphere is difficult, rare and sometimes ethically fraught.
- **The atmosphere is chaotic.** Small perturbations grow unpredictably, so attributing an outcome to a specific intervention is statistically difficult. Cloud-seeding research has struggled with this for decades.
- **Transfer is unproven.** Moving from weather to biology or materials is a big leap. Those domains have different scales, data types and governing laws, and no one has yet shown a single physics model working across them.
- **Nothing is public yet.** Causal Labs hasn't published a model, paper or benchmark, so the causal claims are a research thesis, not a demonstrated result.

In short, "understanding cause and effect" means having a model you can ask "what if?" and trust the answer, not just one that predicts what usually happens next. Weather is the first test because it's the best-measured physical system, and the bigger vision depends on whether that understanding carries over to other physical systems.

## Learning the mechanism
The goal is more about learning the mechanism, meaning **the rules for how one state of the system turns into the next**.

**Root cause vs. mechanism**

In physical systems, effects rarely have a single root cause. A heavy rainfall event comes from many things acting together: ocean temperature, humidity, wind shear, terrain, pressure gradients. Asking "what is *the* cause?" often has no clean answer.

What the model tries to learn is how all those variables push on each other over time, the way a physics engine knows how objects interact. Once it has that, it can answer several kinds of questions:

- **Prediction:** "Given the current state, what happens next?" Run the mechanism forward.
- **Intervention:** "If I change one variable, what happens?" Alter that variable and run the mechanism forward again.
- **Attribution, which is closest to your "root cause" idea:** "Which factors mattered most for this outcome?" Remove or change each factor in turn and see how much the outcome changes.

So root-cause analysis comes out of the model as a byproduct of understanding the mechanism; it isn't the model's primary output.

**Why the mechanism matters for interventions**

A purely correlational model can be badly wrong here. Suppose it learned "dark clouds → rain." If you intervene by painting the sky dark (absurd, but illustrative), it would predict rain, because it never learned that moisture and rising air are what actually produce rain. A model that learned the real mechanism would correctly predict nothing happens.

That's the core bet. If the model has captured the true mechanism and not just surface patterns, its "what if" answers stay reliable even for interventions that never appeared in its training data. If it has only learned correlations, it will forecast well on normal days and fail exactly when you try to change something.

The open question for any AI lab is how to *verify* the model has learned real mechanisms. Good forecasting accuracy alone can't prove that. This is why they're investing in interpretability, to check whether the model's internals respect things like conservation laws, and in comparisons against real-world perturbations.

**The model's knowledge is hidden in its weights**

Even if the model learns a new physical regularity, it stores it implicitly in millions or billions of parameters, not as a neat equation like E = mc². It might predict perfectly without anyone, including the model, being able to state the law. Turning that hidden knowledge into an explicit law takes extra work: **interpretability (probing what the model has learned)** or **symbolic regression (searching for a simple equation that reproduces the model's behavior)**. There's precedent for this (precedent: an action, situation, or decision that has already happened and can be used as a reason why a similar action or decision should be performed or made). Researchers have trained neural networks on physical data and then extracted compact formulas from them, including rediscovering known laws and proposing new empirical formulas in astrophysics. It's real but still small-scale.

**It can only learn what's in its data**

A model can only find laws that leave a footprint in what it observes.


## What's realistic, and still valuable
The more plausible discoveries are **"effective laws"** for messy, complex processes that humans can't derive cleanly from first principles. Examples are how clouds form and produce rain, how turbulence moves heat and moisture, or how aerosols change cloud behavior. Current weather models handle these with **rough approximations**, and they're among the biggest sources of forecast error. A model that learned **(1) better descriptions**, and that **(2) scientists could then extract and understand**, would be a genuine scientific contribution.

So the likely outcome isn't a new Einstein moment. It's AI uncovering hidden rules inside complex systems we already partly understand, and humans then turning those findings into science.

### The core problem: a gap in scale

Weather models divide the atmosphere into a 3D grid. Even the best global models use grid cells several kilometers to around 10 km wide. Within each cell, the model tracks average temperature, wind, humidity and pressure, and steps the physics equations forward in time.

But many important processes happen at much smaller scales:

- **Cloud droplets and ice crystals** are micrometers to millimeters in size. Whether a cloud rains depends on how droplets collide, merge, freeze and fall.
- **Turbulence**, the swirling eddies that mix heat and moisture, happens at scales from meters down to millimeters.
- **Thunderstorm updrafts (a localized upward movement of air or other gas in the atmosphere)** can be narrower than a single grid cell.
- **Aerosols** (dust, sea salt, pollution, smoke) are tiny particles that water condenses on, changing how many droplets form and how big they get.

The model can't see any of this directly. A grid cell might contain half clear sky and half thunderstorm, but the model only knows the averages.

### How current models cope: parameterization

Since these processes can't be simulated directly, scientists write simplified formulas, called parameterizations, that estimate their combined effect from the grid-scale averages. For example: "if humidity in this cell exceeds a threshold and the air is unstable, assume this much convection and this much rain."

These formulas are part physics and part educated guesswork. They contain tuned constants and assumptions that work reasonably on average but fail in specific situations. That's a big reason models get the timing and location of thunderstorms wrong, and why rainfall forecasts are less reliable than temperature forecasts. In climate science, how aerosols affect clouds is one of the largest sources of uncertainty in projecting future warming.

Cloud microphysics studies the fundamental interactions between aerosols, water vapor, liquid drops, and ice particles. While the condensation of water vapor into liquid droplets is well understood, **the phase change from liquid water to ice—known as ice nucleation—remains one of the most complex and least understood areas of atmospheric science**. Pure water in the atmosphere does not freeze at 0°C. It can remain liquid in a "supercooled" state down to approximately -38°C, at which point it freezes homogeneously. For ice crystals to form at warmer sub-zero temperatures (between 0°C and -38°C), the process requires specific microscopic airborne particles to act as seeds, known as Ice Nucleating Particles (INPs).

The effectiveness of an INP depends heavily on its specific molecular structure, surface chemistry, and physical defects. Typical INPs include mineral dust blown from deserts, biological particles (like bacteria, pollen, and fungal spores), and soot from combustion. Because these microscopic surface interactions are incredibly complex and nearly impossible to predict using pure theoretical physics, atmospheric scientists must rely heavily on laboratory cloud chambers and high-altitude air sampling to observe which particles actually trigger freezing under varying environmental conditions.

### DeMott et al. (2010) parameterization

The **DeMott et al. (2010) parameterization** revolutionized how climate models predict ice formation by linking thermodynamic conditions directly to actual atmospheric aerosol measurements.

The formula is expressed as:

$$n_{INP}(T) = a(273.16 - T)^b (n_{aer, >0.5})^{c(273.16 - T) + d}$$

**Where:**

* **$n_{INP}(T)$**: Ice nucleating particle number concentration (per standard liter).
* **$T$**: Cloud temperature in Kelvin.
* **$n_{aer, >0.5}$**: The number concentration (per standard cm³) of aerosol particles with a diameter larger than 0.5 micrometers.
* **$a, b, c, d$**: Empirical constants derived from global field data ($a = 5.94 \times 10^{-5}$, $b = 3.33$, $c = 0.0264$, $d = 0.0033$).

The target of the prediction is exactly the concentration of aerosol particles that successfully catalyze the freezing of water. In atmospheric science, these active particles are called Ice Nucleating Particles (INPs).

The DeMott formula takes the *total* number of large aerosols floating in the air (the input, $n_{aer, >0.5}$) and calculates **the specific, tiny fraction of them that will actually become active ice seeds (the output, $n_{INP}$) at a given temperature**.

Because every single active INP spawns exactly one initial ice crystal—a process known as primary ice nucleation—predicting the number of these triggering aerosols is mathematically identical to predicting the starting number of ice crystals in the cloud. The remaining thousands of other large aerosols in that same volume of air simply remain inactive, either suspended inside liquid droplets without freezing them or floating inertly between them.

#### How is this model important for forecasting rainfall?

The DeMott parameterization is critical for forecasting rainfall because, outside the deep tropics, the vast majority of the world's rain actually begins its life as ice.

To understand why predicting ice dictates the rain forecast, it helps to look at the life cycle of a raindrop:

**The Ice-to-Rain Pipeline**
Clouds are mostly made of supercooled liquid droplets that are too light to fall against the updrafts of air. When a specific aerosol triggers an ice crystal (the exact event DeMott predicts), a physical mechanism called the Wegener-Bergeron-Findeisen process takes over. Because the vapor pressure over ice is lower than over liquid, the newly formed ice crystal acts like a magnet, rapidly stealing moisture from the surrounding liquid droplets.

**Falling to Earth**
As the crystal hoovers up this moisture, it quickly grows into a heavy snowflake and falls out of the cloud. As it drops through the warmer lower atmosphere, it melts into a raindrop. If no ice forms, the cloud droplets simply stay suspended—meaning no rain.

**Making or Breaking the Forecast**
If a weather model gets the INP count wrong, the precipitation forecast fails completely:

* **Underpredicting INPs:** The model thinks the cloud stays entirely liquid. Since the droplets never grow heavy enough to fall, the forecast predicts a cloudy day, completely missing an impending rainstorm.
* **Overpredicting INPs:** The model assumes thousands of ice crystals form. The available cloud moisture gets divided among so many crystals that none of them grow large enough to fall. The forecast might predict drizzle or nothing at all, missing the dynamics of the actual storm.

By using physical dust and aerosol measurements to get that initial ice crystal count right, the DeMott model ensures weather supercomputers correctly predict when a cloud will finally break and drop its water on the ground.

### Why first principles aren't enough

For turbulence, the fundamental equations (fluid dynamics) are known. The problem is that solving them at every scale for the whole planet would take astronomically more computing power than exists.

For cloud microphysics, the fundamentals are only partly known. How ice crystals form, for example, depends on which particles act as seeds, and that's still being worked out experimentally.

Identifying the exact mechanisms of heterogeneous ice nucleation is still an active frontier of experimental science, and this incomplete understanding remains a major source of uncertainty in global weather and climate models.

Either way, nobody can derive the right grid-scale formula cleanly. What's needed is an "effective law": a rule describing how the small-scale chaos behaves on average at the larger scale.

### Effective laws are common in physics

Much of everyday physics consists of effective laws:

- The **ideal gas law** (pressure, volume, temperature) describes trillions of molecules without tracking any of them.
- **Ohm's law** describes current through a wire without modeling individual electrons.
- **Friction and drag formulas** summarize enormously complicated surface and fluid interactions.

Each of these was discovered empirically before it was fully explained from deeper theory. Cloud and turbulence processes need the same kind of law, but they're too complex for humans to find by inspection of data.

### Where AI could help

There are three ways AI could contribute:

1. **Learning the effective law.** Researchers run extremely high-resolution simulations of small regions, or gather detailed observations, then train a model to predict what the coarse grid should "feel" from the small-scale processes. The model learns the effective law implicitly. Researchers have already shown this works for convection and ocean turbulence in research settings.
2. **Extracting it into human form.** The learned relationship can then be converted into an explicit equation using techniques like symbolic regression, which searches for compact formulas that reproduce the model's behavior. There have been early successes, including compact equations for how ocean eddies transfer momentum that scientists could examine and interpret.
3. **Scientists explaining it.** Once you have a candidate equation, physicists can ask why it has that form, which deeper processes produce it, and when it should break down. That's where it becomes science, not just a better forecast.

A large model like LPM, trained on huge amounts of real observations rather than one simulation, could in principle pick up regularities (consistent patterns in nature) that neither current formulas nor small simulations capture.

**The difficulties**

- **Stability:** learned components plugged into a weather or climate model can push the simulation into unrealistic states over long runs.
- **Extrapolation:** a law learned from today's climate may not hold in a warmer one, which is exactly where climate projections need it.
- **Physical consistency:** a learned rule might quietly violate conservation of energy or water, producing plausible short-term forecasts while being physically wrong.
- **Extraction may fail:** the model's knowledge might not reduce to a simple formula. You might get accuracy without insight.

**A Kepler moment, not an Einstein moment**

The best historical comparison is Kepler. Working from Tycho Brahe's careful astronomical observations, Kepler found that planets move in ellipses and follow precise mathematical patterns. These were empirical laws, true but unexplained. Decades later, Newton explained them with gravity.

AI's likely role is similar to Kepler's: **(1) finding accurate, compact rules hidden in huge amounts of data**. **(2) Humans then play Newton's part and explain why those rules hold**. That would be a modest kind of discovery compared with relativity, but better rules for clouds, turbulence and aerosols would improve rainfall forecasts, severe-storm warnings and climate projections, which matters a great deal in practice.

### Why does it matter a great deal in practice?

Because rain, storms and climate are where weather affects lives, money and long-term planning most, and they're exactly the parts current models handle worst.

**1. Flooding and water**

Temperature forecasts are already quite accurate days ahead. Rainfall forecasts are much weaker, especially for intense, localized downpours (a lot of rain in a short time) from thunderstorms. Those are driven by convection and cloud processes, the very things models approximate poorly.

That matters because:
- **Flash floods** can kill people and damage property within hours, often from storms that models placed in the wrong spot or underestimated.
- **Reservoirs and drainage** are run on rainfall forecasts. Operators decide when to release water or prepare pumps based on how much rain is coming and where.
- **Tropical regions** get much of their rain from thunderstorms, so convection errors hit them hardest.

**2. Severe-storm warnings save lives**

Better warnings give people time to evacuate, secure property and move equipment. The benefits are well documented: studies of early warning systems consistently find that even a day's notice substantially reduces deaths and damage. Improving the *precision* of warnings also matters. Warnings that are too broad or frequently wrong lead people to ignore them, and more accurate forecasts restore trust.

Hurricane and typhoon intensity is a good example. Forecasts of where a storm will go have improved greatly over recent decades, but forecasts of how strong it will get, especially rapid intensification, lag behind, partly because they depend on small-scale convection and ocean-atmosphere mixing.

**3. The economy runs on weather**

Many industries make daily decisions based on forecasts:
- **Agriculture:** when to plant, irrigate, spray and harvest. A wrong rain forecast can waste fertilizer or ruin a harvest.
- **Energy:** solar output depends on cloud cover, and grid operators need to balance supply and demand hour by hour. As more power comes from solar and wind, cloud and wind forecast errors become grid reliability problems.
- **Aviation and shipping:** routing around storms and turbulence affects safety, fuel and delays.
- **Insurance:** pricing risk depends on how well extreme events can be anticipated.

Small accuracy gains, multiplied across millions of decisions, add up to large economic value. This is the near-term market Causal Labs is targeting.

**4. Climate projections guide decisions lasting decades**

This may be the biggest stake. How clouds respond to warming is one of the main reasons scientists can't pin down how much the planet will warm for a given amount of emissions. Aerosol-cloud interactions add further uncertainty about how much warming pollution has been masking.

That uncertainty flows into real decisions:
- How high to build sea walls and flood defenses.
- How to design drainage, water supply and power systems meant to last 50 years.
- How to plan agriculture as rainfall patterns shift.
- How urgently emissions need to fall to stay within temperature targets.

Narrowing that uncertainty would let governments and businesses plan with more confidence, avoiding both under-preparing and spending heavily on protection that turns out unnecessary.

**In short**

The parts of weather that matter most to people, such as heavy rain, severe storms and long-term climate change, depend on small-scale processes that models currently approximate. Better rules for those processes would improve forecasts where the stakes are highest.