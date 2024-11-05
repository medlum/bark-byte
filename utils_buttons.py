
import streamlit_antd_components as sac


def select_tab():
    return sac.tabs([
        sac.TabsItem(label='chat', tag="main"),
        sac.TabsItem(label='schedule', icon='calendar-day'),
        sac.TabsItem(label='health', icon='lungs-fill'),
        sac.TabsItem(label='disabled', disabled=True),
    ], align='center',
        position='top',
        size='sm',
        use_container_width=True,
        index=0,
    )
