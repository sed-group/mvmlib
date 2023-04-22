Background
####################

The :term:`MVM` developed by :cite:author:`Brahma2020` provides a systematic process for analyzing and evaluating design alternatives at the preliminary design stage in terms of the value of design margins. :term:`MVM` is an analytical method that uses dependencies in a design to first analyse it, then localise the excess margin and finally quantify them. The quantification is done considering the context of the desirable aspects of margin such as change absorption potential and the undesirable effect margins have on the performance of the system. Change absorption is quantified by incrementally varying the input requirements and calculating the maximum amount of deviation in the requirement that can be supported by the current design. The impact of margins on the product's performance is computed by calculating the loss in performance relative to a hypothetical design featuring zero margins, denoted as the threshold design.


Overall, :term:`MVM` is structured in the following steps:

* Construct a Margin Analysis Network: The first step is to construct the margin analysis network (MAN). The network identifies the flow of information from the input (specification) parameters through decisions to the performance parameters. For some companies, such relationship networks may already exist for a design, which can be reused or adapted to be used in :term:`MVM`. The following steps briefly describe how to create a MAN:

    * Model the design process: The first step, which shows the possible arrangement of tasks needed for the design to be completed. The aim of this step is to identify the objectives of the design, the individual tasks and subtasks and more importantly, the input (specification) (:math:`\mathbf{s} \in \mathbb{R}^n`) and performance parameters (:math:`\mathbf{p} \in \mathbb{R}^m`) of interest.
    * Identify steps and parameters: For each tasks or subtasks, the individual inputs, outputs, decisions and calculations are identified.
    * Connect key parameters into a network: Key parameters are identified and unnecessary parameters are eliminated. Inputs and outputs of tasks are connected to each other for all the systems /subsystems to form a network.
    * Construct MAN: Once the network is formed, the margin analysis network can be created. This can be done by modelling the detailed information flows within the calculation and decision steps. The decision steps involve deciding on off-the-shelf components, i.e., choosing from a discrete set :math:`\mathcal{D}`. For example, the decision for a material :math:`d \in \left\{\text{Inconel 718, Titanium}\right\}`. The :term:`MAN` is a collection of vector-valued functions. 

    .. admonition:: Definition: Margin analysis network (:term:`MAN`)
        :class: definitions

        * The function :math:`\mathbf{f}: \mathbb{R}^{n}\times\mathcal{D} \to \mathbb{R}^{e}` takes input specifications :math:`\mathbf{s}` and decisions :math:`\mathbf{d}` as its arguments and returns the corresponding target thresholds :math:`\mathbf{e}^t`.
        * The function :math:`\mathbf{g}: \mathcal{D} \to \mathbb{R}^{e}` takes decisions :math:`\mathbf{d}` as its arguments and returns the corresponding decided values :math:`\mathbf{e}^d`.
        * The function :math:`\mathbf{h}: \mathbb{R}^{n}\times\mathcal{D} \to \mathbb{R}^{m}` takes input specifications :math:`\mathbf{s}` and decisions :math:`\mathbf{d}` as its arguments and returns the corresponding performance parameters :math:`\mathbf{p}`.
        * The function :math:`\mathbf{g}^{-1}: \mathbb{R}^{e} \to \mathcal{R}^{e}` takes decided values :math:`\mathbf{e}^d` as its arguments and returns the corresponding threshold design :math:`\mathbf{d}^\text{threshold}`.

* Margin Value Analysis: The margin value analysis consists of calculating three metrics:   

    * Local excess margin: The first metric quantifies the local excess at each margin node in the margin analysis network. In other words it quantifies by how much a :term:`DV` exceeds the required :term:`TT` at each margin node. This metric is based on the definition of excess by :cite:author:`Eckert2019`.
    
    .. admonition:: Definition: Excess margins
        :class: definitions

        .. math::
            :label: eq:excess

            \mathbf{e} = \left(\mathbf{f}(\mathbf{s},\mathbf{c}) - \mathbf{g}(\mathbf{c})\right){\mathbf{i}^\text{excess}}^\textsf{T} = \left(\mathbf{e}^t - \mathbf{e}^d\right){\mathbf{i}^\text{excess}}^\textsf{T},
        
        where the vector :math:`\mathbf{i}^\text{excess}` is defined as 
        
        .. math::
            :label: eq:iopsign1

            i_e^\text{excess}={\begin{cases}1&{\text{if }t_e \text{ must exceed }d_e}\\-1&{\text{if }t_e \text{ must not exceed }d_e}\end{cases}}.
    
    .. pcode::
        :linenos:    

        \begin{algorithm}
        \caption{Calculation of MAN parameters}
        \begin{algorithmic}
        %
        \PROCEDURE{forward}{$\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$, $\mathbf{s}$, $\mathbf{d}$, $\mathbf{i}^\text{excess}$}
            \STATE $\mathbf{e}^t = \mathbf{f}(\mathbf{s},\mathbf{c})$
            \STATE $\mathbf{e}^d = \mathbf{g}(\mathbf{c})$
            \STATE $\mathbf{e} = \left(\mathbf{e}^t - \mathbf{e}^d\right){\mathbf{i}^\text{excess}}^\textsf{T}$
            \STATE $\mathbf{p} =  \mathbf{h}(\mathbf{s},\mathbf{c})$
        \ENDPROCEDURE
        %
        \end{algorithmic}
        \end{algorithm}

    * :term:`IoP`: The second metric is calculated based on the performance parameters -- considering the actual design versus the performance parameters -- if it is assumed that the margin is eliminated at each node. Elimination of the :math:`i` th margin can be done by substituting a vector :math:`\mathbf{e}_i^d` of decided values, where the :math:`i` th component :math:`e^d_i` is set equal to the :math:`i` th threshold value, into a function :math:`\mathbf{g}^{-1}: \mathbb{R}^{e} \to \mathcal{R}^{e}` to obtain the *threshold design* :math:`\mathbf{d}_i^\text{threshold} = \mathbf{g}^{-1}(\mathbf{e}^d_i)`. In this paper, :math:`\mathbf{g}^{-1}` is found using surrogate modeling of :math:`\mathbf{g}` on the set :math:`\mathcal{D}`. The impact may then be calculated by calculating the relative difference in performance between each threshold design and the nominal chosen design.
    
    .. admonition:: Definition: Impact on performance (:term:`IoP`)
        :class: definitions

        .. math::
            :label: eq:iop

            \begin{aligned}
                I_{i,j} & = \dfrac{h_j(\mathbf{s},\mathbf{d}) - h_j(\mathbf{s},\mathbf{d}_i^\text{threshold})}{h_j(\mathbf{s},\mathbf{d}_i^\text{threshold})}{i_j^p} \\
                & = \dfrac{p_j - p_{j,i}^\text{threshold}}{p_{j,i}^\text{threshold}}{i_j^p},
            \end{aligned}
    
        where :math:`p_j` is the :math:`j`th performance parameter, :math:`p_{j,i}^\text{threshold}` is the :math:`j`th performance parameter of the :math:`i` th threshold design, and :math:`\mathbf{i}^p` is sign of each performance parameter.
        
        .. math::
            :label: eq:iopsign2

            i_j^p={\begin{cases}1&{\text{if increasing }p_j \text{ is valuable}}\\-1&{\text{if decreasing }p_j \text{ is valuable}}\end{cases}}.

    .. pcode::
        :linenos:    

        \begin{algorithm}
        \caption{Impact on performance}
        \begin{algorithmic}
        %
        \PROCEDURE{impact}{$\mathbf{g}^{-1}$, $\mathbf{h}$, $\mathbf{e}^t$, $\mathbf{e}^d$, $\mathbf{e}$, $\mathbf{p}$, $\mathbf{s}$, $\mathbf{i}^p$}
            \FOR{$i = 1, 2, ..., e$}
                \STATE $\mathbf{d}_i^\text{threshold} = \mathbf{g}^{-1}({e}^{t}_{i},{e}^{d}_{\neq i})$
                \STATE $\mathbf{p}_i^\text{threshold} = \mathbf{h}(\mathbf{s},\mathbf{d}_i^\text{threshold})$
                \STATE $\mathbf{I}_{i} = \left(\mathbf{p} - \mathbf{p}_{i}^\text{threshold}\right) \oslash \left({\mathbf{p}_{i}^\text{threshold}} \circ {\mathbf{i}^p}\right)$
            \ENDFOR
        \ENDPROCEDURE
        %
        \end{algorithmic}
        \end{algorithm}

    This metric indicates the deteriorating effect of each margin node on each performance parameter.

    * :term:`CAC`: The third metric is calculated by first calculating the variance in the specification possible which do not lead to any changes in the design and then quantifying the effect of that on the margin node. In other words, how much of the margin at each margin node can actually be absorbed without causing a propagation of changes. The allowable variance in the specifications is calculated as follows:
    
    .. admonition:: Definition: Deterioration of an input specification
        :class: definitions

        .. math::
            :label: eq:speclimit

            v_k=\dfrac{s_k^\text{max} - s_k}{s_k}{i_j^s},
    
        where :math:`s_k^\text{max}` is the value of the :math:`k` th specification (while setting all others to their nominal values) when any margin :math:`\mathbf{e}` is less than or equal to zero. We call this value the \text{maximum supported} specification. :math:`\mathbf{i}^s` is the sign of the expected change in each specification parameter.
            
        .. math::
            :label: eq:iopsign3

            i_k^s={\begin{cases}1&{s_k \text{ is expected to increase}}\\-1&{s_k \text{ is expected to decrease}}\end{cases}}.
        
    This value is found by performing a line search along each specification. 
    
    .. pcode::
        :linenos:    

        \begin{algorithm}
        \caption{Maximum allowable deterioration in input specifications}
        \begin{algorithmic}
        %
        \PROCEDURE{limit}{$\mathbf{f}$, $\mathbf{g}$, $\mathbf{d}$, $\mathbf{c}$, $\mathbf{s}$, $\mathbf{i}^s$, $\boldsymbol{\Delta}^s$, $k$}
            \WHILE{$\min{\left(\mathbf{f}(\hat{s}_k,s_{\neq k},\mathbf{c}) - \mathbf{g}(\mathbf{d})\right){\mathbf{i}^\text{excess}}^\textsf{T}} < \mathbf{0}$}
                \STATE $\hat{s}_k \gets \hat{s}_k + i_k^s\Delta_k^s$
            \ENDWHILE
            \STATE $s_k^\text{max} \gets \hat{s}_k$
        \ENDPROCEDURE
        %
        \end{algorithmic}
        \end{algorithm}

    The next step is to calculate the portion of absorbed margin for each specification by calculating the relative change in the threshold values.
    
    .. admonition:: Definition: Change absorption capability (:term:`CAC`)
        :class: definitions

        .. math::
            :label: eq:cac1

            \begin{aligned}
                A_{i,k} & = \dfrac{f_i(s_k^\text{max},s_{\neq k},\mathbf{d}) - f_i(\mathbf{s},\mathbf{d}) }{ f_i(\mathbf{s},\mathbf{d})v_k} \\
                & = \dfrac{e^{t,\text{max}}_{k,i} - e^t_i}{{e^t_i}{v_k}},
            \end{aligned}
        
        where :math:`t_i` is the :math:`i` th threshold parameter, :math:`e^{t,\text{max}}_{k,i}` is the :math:`i` th threshold parameter corresponding to the :math:`k` th maximum supported specification.

    .. pcode::
        :linenos:    

        \begin{algorithm}
        \caption{change absorption capability}
        \begin{algorithmic}
        %
        \PROCEDURE{absorption}{$\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$, $\mathbf{d}$, $\mathbf{c}$, $\mathbf{s}$, $\mathbf{i}^s$}
            \STATE $\mathbf{e}^{d,\text{threshold}} \gets \mathbf{e}^d$, $e^{d,\text{threshold}}_i \gets e^t_i$
            \STATE $\mathbf{d}_i^\text{threshold} = \mathbf{g}^{-1}(\mathbf{e}^{d,\text{threshold})}$
            \STATE $\mathbf{p}_i^\text{threshold} = \mathbf{h}(\mathbf{s},\mathbf{d}_i^\text{threshold})$
            \STATE initialize vector of maximum supported specifications $\mathbf{s}^\text{max} \in \mathbb{R}^n$
            \FOR{$k = 1, 2, ..., n$}
                \STATE $s_k^\text{max} = \text{LIMIT}(\mathbf{f}, \mathbf{g}, \mathbf{d}, \mathbf{c}, \mathbf{s}, \mathbf{i}^s, k)$
                \STATE $v_k=\dfrac{\left({s}_k^\text{max} - {s}_k\right)}{\mathbf{i}^s}{\mathbf{s}}$
                \STATE $\mathbf{A}_{k} \gets \left(\mathbf{f}(s_k^\text{max},s_{\neq k},\mathbf{d}) - \mathbf{f}(\mathbf{s},\mathbf{d}) \right) \oslash \left({v_k}\mathbf{f}(\mathbf{s},\mathbf{d})\right)$
            \ENDFOR
        \ENDPROCEDURE
        %
        \end{algorithmic}
        \end{algorithm}

* Aggregating metrics: The method suggests that the metrics calculated for impact and absorption be averaged based on the relative importance of the the performance parameters and the prioritisation of change in the specification. This leads to aggregate values of the two metrics. In this paper we assume equal weightings for all performance parameters and input specifications. The aggregation can be performed by taking the mean across the :math:`i` th direction for absorption and impact as follows:

.. math::
    :label: eq:cac2

    \begin{aligned}
        \overline{A}_i & = \dfrac{1}{n}\sum_{k=1}^{n} a_{i,k} \\
        \overline{I}_i & = \dfrac{1}{m}\sum_{j=1}^{m} i_{i,j} \\
    \end{aligned}

* :term:`MVP`: The two metrics (:math:`\overline{\mathbf{A}}` and :math:`\overline{\mathbf{I}}`) can then be plotted which shows the relative importance of the :math:`i` th margin node in terms of their absorption capability and the deteriorating effect on the performance of the system.

.. note::
    :term:`MVM` is based on the idea that excess margins are created when decisions are made in a design process. The method therefore provides guidance towards identifying those decisions, leading to the identification of all the margins in a design.  The method, in its original form, was specifically developed for an incremental design context, where a design already exists and improvements are supposed to be made by making minor adjustments to the design. The method also focuses extensively on off-the-shelf components, use of which is a common practice in routine design situations. However, for the context of this paper, designs do not pre-exist and are generated through a functional modelling based :term:`DSE` approach. 
    The :ref:`examples <examples>` provided in the documentation of this library are of a highly-integrated with continuous and discrete off-the-shelf parts showing how the :term:`MVM` can be adapted for :term:`DSE` studies.

.. pcode::
   :linenos:    

    \begin{algorithm}
    \caption{Pseudo-algorithm for the MVM}
    \begin{algorithmic}
    %
    \PROCEDURE{stochasticMVM}{$F_\mathbf{S}(\mathbf{s})$}
        \STATE Initialization:
        \STATE set counter $i_\text{sample} \gets 0$
        \STATE initialize aggregate impact and absorption $\overline{\overline{\mathbf{I}}} \in \mathbb{R}^e$ and $\overline{\overline{\mathbf{A}}} \in \mathbb{R}^e$
        \WHILE{$i_\text{sample} \leq n_\text{samples}$}
            \STATE draw sample input specifications $\mathbf{S}\in\mathbb{R}^n \sim F_\mathbf{S}(\mathbf{s})$
            \STATE $\mathbf{e}^t, \mathbf{e}^d, \mathbf{e}, \mathbf{p} = \text{FORWARD}(\mathbf{f}, \mathbf{g}, \mathbf{h}, \mathbf{S}, \mathbf{d}, \mathbf{i}^\text{excess})$
            \IF{$\exists~i \text{ such that } e_i > 0$}
                \STATE set $i_\text{sample} \gets i_\text{sample} + 1$ 
                \CONTINUE
            \ENDIF
            \STATE $\mathbf{I} = \text{IMPACT}(\mathbf{g}^{-1}, \mathbf{h},\mathbf{e}^t,\mathbf{e}^d,\mathbf{e},\mathbf{p},\mathbf{S},\mathbf{i}^p)$
            \STATE $\mathbf{A} = \text{ABSORPTION}(\mathbf{f}, \mathbf{g}, \mathbf{h}, \mathbf{d}, \mathbf{c}, \mathbf{S}, \mathbf{i}^s)$
            \STATE $\overline{\mathbf{A}} = \dfrac{1}{n}\sum\limits_{k=1}^{n} \mathbf{A}_k$, 
            $\overline{\mathbf{I}} = \dfrac{1}{m}\sum\limits_{j=1}^{m} \mathbf{I}_j$
            \STATE $\overline{\overline{\mathbf{A}}} \gets \overline{\overline{\mathbf{A}}} + \overline{\mathbf{A}}$, 
            $\overline{\overline{\mathbf{I}}} \gets \overline{\overline{\mathbf{I}}} + \overline{\mathbf{I}}$
            \STATE $i_\text{sample} \gets i_\text{sample} + 1$
        \ENDWHILE
        \STATE $\overline{\overline{\mathbf{A}}} \gets \overline{\overline{\mathbf{A}}}/n_\text{samples}$ 
        \STATE $\overline{\overline{\mathbf{I}}} \gets \overline{\overline{\mathbf{I}}}/n_\text{samples}$
    \ENDPROCEDURE
    %
    \end{algorithmic}
    \end{algorithm}

.. I need to refer to this :eq:`eq:cac2`
.. \mathbf{e}^t, \mathbf{e}^d, \mathbf{e}, \mathbf{p}