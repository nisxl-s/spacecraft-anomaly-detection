# ESA Mission 1 mission-ticket escalation

This development replay uses sealed residual detector output for six channels with calendar-year reporting for 2009–2013. A mission ticket opens and pages on the first alarm, groups simultaneous starts, and appends all subsequent alarm starts. A ticket closes at a coverage gap or after 24 consecutive eligible, alarm-free hours. The selected rule adds at most one escalation page when three distinct channels have started alarm episodes during the current hour and the preceding five hourly bins. The original threshold and detector emissions are unchanged.

A page receives credit only for alarm episodes starting at that page timestamp. Later updates remain available as ticket evidence, without assuming operator review. The audit reconstructs the page timestamps independently and verifies full preservation of the source alarms in the tickets. The matching implementation and causal telemetry preparation are reused from the prior frozen ESA studies.

The selected rule passed development gates. Selection used the same three development incidents as the grid evaluation. Separate compatible labeled telemetry is required for independent validation. The existing six files end in 2013. The longest development ticket lasted 1,577 hours and can issue only one escalation.

The archive includes the baseline and all six grid results, selected page and ticket ledgers, audit, and frozen residual weights, transforms and thresholds. The execution scripts reference the preceding ESA pilot and frozen replay workspaces.
